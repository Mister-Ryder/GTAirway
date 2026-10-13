# NaviAirway — earlier research version

This branch preserves the earlier NaviAirway-based airway segmentation code as a code-only snapshot. The main branch contains the later GTAirway experiments.

## Code map

- train.py: supervised CT airway segmentation baseline.
- train_semi_supervised_learning.py: teacher/student experiment.
- Get_dataset_info.py and BAS_*: patch statistics and sampling experiments.
- NaviAirway_pipeline.py: sliding-window inference and tree post-processing.
- func/: models, losses, data loading and post-processing.

The training data, checkpoints, generated metadata, DICOM, NIfTI and experiment outputs are deliberately excluded. Existing scripts contain historical absolute paths and have not been validated as an end-to-end runnable release. See requirements.txt as a historical dependency list, not a locked environment.

## Provenance

This version builds on NaviAirway by Anton Wang. The original MIT license and copyright notice are preserved in [LICENSE](LICENSE). The source README referenced a personal download link and password; those credentials are not part of this repository.
