# Revision v3 Artifact Directory

This directory is the canonical destination for revision-v3 experiment outputs.

Only JSON result files in this directory should be used for final manuscript
tables. Failed exploratory runs and older non-v3 result files have been removed
so this directory contains only usable revision-v3 artifacts.

Adversarial reconstruction artifacts are stored separately from task-performance
artifacts:

- `adversarial_results.json`: combined adversarial benchmark summary.
- `adversarial-*_results.json`: per-base-experiment attack metrics.
- `horio-adversary-comparison-*_results.json`: black-box, gray-box, and
  white-box comparison against the Horio et al. restricted permutation baseline.
- `permutation-family-key-recovery-comparison_results.json`: direct
  obfuscated-image quality and known/chosen-plaintext recovery comparison for
  Kiya/Kinoshita-family permutation, EtC, restricted-permutation, and related
  reversible secret-key baselines.
- `figures/adversarial-*_grid.png`: qualitative reconstruction grids.
- `figures/horio-adversary-comparison-*_grid.png`: qualitative Horio baseline
  reconstruction/comparison grids.
- `figures/permutation-family-key-recovery-comparison_grid.png`: qualitative
  obfuscated-image grid for the broader permutation-family comparison.
