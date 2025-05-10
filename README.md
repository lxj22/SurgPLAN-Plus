# SurgPLAN++

## Dataset Generation

## Feature Extractor
we finetune the SlowFast as our feature extractor.

1. run feature_extract/extract_feature.py to get the extracted feature for next step training.
You need to change the   ```checkpoint = torch.load("trained_slowfast_model.pyth")``` in line 100 to your finetuned slowfast model path.

2. You need to modify the config file ```SurgPLAN-Plus/feature_extract/configs/SLOWFAST_8x8_R50.yaml```.
modify the config file, change ```PATH_TO_DATA_DIR``` to your your dataset folder. In the folder, you should have your training videos and one csv file containing video names to be processed.
For example:
```
video01.mp4
video02.mp4
......
```
Also modify ```OUTPUT_DIR``` to your output feature folder.

3. run ```python extract_feature.py --cfg ./configs/SLOWFAST_8x8_R50.yaml```

## Train the SurgPLAN framework

1. before run training. Modify the config file in ```SurgPLAN-Plus/configs/cataract_slowfast.yaml```
   change ```json_file``` to your dataset json file,
   change ```feat_folder``` tp your feature folder as extracted in Feature Extractor section


