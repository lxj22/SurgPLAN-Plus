# SurgPLAN++

## Dataset Generation

You need to create dataset in data/your_dataset_name as follows: 
```
{
    "train01.mp4": [
        {
            "start": 1.4,
            "end": 16.4,
            "label": 2,
            "duration": 15.999999999999998,
            "subset": "training",
            "time_till_now": 16.4
        },
        {
            "start": 31.466666666666665,
            "end": 35.36666666666667,
            "label": 3,
            "duration": 4.900000000000002,
            "subset": "training",
            "time_till_now": 35.36666666666667
        },
......
```
A dictionary contains video name as key, value is a list contains each phase segmentation. Inside one segmentation, there is a start time, end time, phase label, duration, subset(training or testing), time_till_now(equals end time).

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
   change ```num_classes``` to your phase classes plus one(idle phase).

2. Run ```python train.py --cfg cataract_slowfast.yaml``` to train backbone.


## Online inference

1. run ```online_surgplan_inference.py```

