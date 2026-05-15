import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from vit_obfuscation.config.experiment import DatasetConfig, ExperimentConfig
from vit_obfuscation.datasets.loader import _resolve_hf_dataset_name
from vit_obfuscation.obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from vit_obfuscation.tasks.anomaly import _load_guided_anomaly_dataset
from vit_obfuscation.tasks.base import BaseTask
from vit_obfuscation.tasks.feature_utils import image_to_binary_mask
from vit_obfuscation.tasks.object_detection import (
    _default_detection_id2label,
    _label_id_map,
    _normalize_detection_labels,
    ObjectDetectionTask,
)


def test_resolve_scene_parse_dataset_name():
    assert _resolve_hf_dataset_name("scene_parse_150") == "zhoubolei/scene_parse_150"
    assert _resolve_hf_dataset_name("coco") == "coco"


def test_normalize_detection_labels_xyxy_to_xywh():
    labels = [
        {
            "bbox": [1.0, 2.0, 5.0, 6.0],
            "category": 3,
            "area": 16.0,
        }
    ]
    normalized = _normalize_detection_labels(
        labels, image_ids=[7], image_sizes=[(10, 10)]
    )

    assert normalized == [
        {
            "image_id": 7,
            "annotations": [
                {
                    "bbox": [1.0, 2.0, 4.0, 4.0],
                    "category_id": 3,
                    "area": 16.0,
                }
            ],
        }
    ]


def test_normalize_detection_labels_handles_coco_segmentation_area():
    labels = [
        {
            "bbox": [236.98, 142.51, 261.68, 212.01],
            "category": 58,
            "area": 531.8071000000001,
        }
    ]

    normalized = _normalize_detection_labels(
        labels,
        image_sizes=[(640, 426)],
        box_format="xyxy",
    )
    bbox = normalized[0]["annotations"][0]["bbox"]

    assert bbox == [236.98, 142.51, 24.700000000000017, 69.5]


def test_normalize_detection_labels_handles_strings():
    labels = json.dumps(
        [
            {
                "bbox": [1.0, 2.0, 5.0, 6.0],
                "category": 3,
                "area": 16.0,
            }
        ]
    )
    normalized = _normalize_detection_labels(labels, image_ids=[7], image_sizes=[(10, 10)])
    assert normalized[0]["image_id"] == 7
    assert normalized[0]["annotations"][0]["category_id"] == 3


def test_coco_zero_shot_label_fallback():
    id2label = _default_detection_id2label("detection-datasets/coco")
    assert id2label is not None
    assert id2label[0] == "person"
    assert id2label[79] == "toothbrush"


def test_coco_80_labels_map_to_yolos_91_labels():
    source = {0: "person", 1: "bicycle", 11: "stop sign", 79: "toothbrush"}
    target = {
        0: "N/A",
        1: "person",
        2: "bicycle",
        12: "N/A",
        13: "stop sign",
        90: "toothbrush",
    }

    assert _label_id_map(source, target) == {0: 1, 1: 2, 11: 13, 79: 90}


def test_object_detection_processor_kwargs_matches_obfuscator_image_size():
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=(512, 512), num_channels=3, patch_size=16, group_size=1
    )
    task = ObjectDetectionTask(
        adapter=SimpleNamespace(),
        obfuscator=obfuscator,
        obf_embedding=SimpleNamespace(),
        processor=SimpleNamespace(),
        config=SimpleNamespace(),
    )

    assert task._get_processor_kwargs() == {"size": {"height": 512, "width": 512}}


def test_guided_anomaly_dataset_builds_bounded_normal_and_anomaly_rows(monkeypatch):
    normal = Image.new("RGB", (4, 4), "black")
    anomaly = Image.new("RGB", (4, 4), "white")
    mask = Image.new("L", (4, 4), 255)

    def fake_load_dataset(*args, **kwargs):
        assert args[0] == "Kelvin878/mvtec"
        assert kwargs["split"] == "train[:2]"
        return [{"image": anomaly, "guide": normal, "mask_image": mask}]

    monkeypatch.setattr(
        "vit_obfuscation.tasks.anomaly.datasets.load_dataset",
        fake_load_dataset,
    )
    config = ExperimentConfig()
    config.dataset = DatasetConfig(
        hf_dataset_name_or_path="Kelvin878/mvtec",
        input_column="image",
        label_column="label",
        mask_column="mask_image",
        normal_column="guide",
        normal_label=0,
    )

    records = _load_guided_anomaly_dataset(
        config,
        split="train[:2]",
        max_samples=2,
        train_normals_only=False,
    )

    assert [row["label"] for row in records] == [0, 1]
    assert image_to_binary_mask(records[0]["mask_image"]).sum().item() == 0
    assert image_to_binary_mask(records[1]["mask_image"]).sum().item() == 16


def test_base_task_checkpoint_round_trip():
    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__(
                adapter=SimpleNamespace(model=nn.Linear(2, 2)),
                obfuscator=SimpleNamespace(),
                obf_embedding=SimpleNamespace(),
                processor=SimpleNamespace(),
                config=SimpleNamespace(),
            )
            self.head = nn.Linear(2, 1)

        def forward(self, images, with_obfuscation=False, **kwargs):
            return images

        def train_task(self):
            return None

        def evaluate(self, with_obfuscation=False):
            return None

    source = DummyTask()
    with torch.no_grad():
        source.adapter.model.weight.fill_(1.5)
        source.head.bias.fill_(2.0)
    state = source.task_state_dict()

    target = DummyTask()
    target.load_task_state_dict(state)

    assert torch.equal(target.adapter.model.weight, source.adapter.model.weight)
    assert torch.equal(target.head.bias, source.head.bias)
