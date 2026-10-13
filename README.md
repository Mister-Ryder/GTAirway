# GTAirway

Research code for CT airway segmentation with patch balancing, anatomical priors and condition-guided fine airway segmentation.

## Branches

- main: later GTAirway research version (this branch).
- legacy/naviairway: earlier NaviAirway-based version.

Both branches are code-only snapshots of historical experiments. They do not include CT/DICOM/NIfTI data, generated patch arrays, metadata pickle files, checkpoints, or experiment outputs. The code has passed Python syntax parsing but has not been verified as an end-to-end runnable release.

## Main code map

- dataset_pre/AirwayMetadataGenerator.py: patch-level generation, structure and location labels derived from external airway-tree parsing results.
- func/AnatomicalClassifier.py: three anatomical classification heads and guidance features.
- func/EnhancedAirwaySegmentation2gau.py and func/AnatomicalFusionModules.py: condition-guided segmentation and Gaussian auxiliary output.
- resume_train_enhanced_airway_DDP_gaumse.py: the later script whose header calls it the fixed training version.
- func/model_arch_model10_plus_tou.py and training-script3.py: an earlier cluster-conditioned experiment.
- evaluate_skeleton.py: historical evaluation experiment.

Historical run notes and the older anatomical guide are preserved under docs/. They are not guaranteed to match the later scripts.

## Reproduction status

This snapshot depends on external BAS images, labels, parsed airway-tree outputs, cropped Gaussian maps and pretrained checkpoints. Some historical paths are hard-coded. The current classifier data loader, distributed sampler and evaluation entry point require fixes before this version can be reproduced end to end. No clinical or patient data should be committed to this repository.

## Provenance

Parts of the code descend from NaviAirway by Anton Wang. The original MIT copyright and license are preserved in [licenses/NaviAirway-MIT.txt](licenses/NaviAirway-MIT.txt); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No license is asserted here for the remaining GTAirway research code.
