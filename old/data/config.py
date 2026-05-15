import yaml

from dataclasses import dataclass, field

from typing import Optional

@dataclass
class Config:
    description: Optional[str] = field(kw_only=True, default=None)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(**config)
