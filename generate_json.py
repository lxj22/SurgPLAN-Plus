import pandas as pd
import json
import os
import numpy as np

df_train = pd.read_csv('train_csv')
df_test = pd.read_csv('test_csv')

df_all = pd.concat([df_train,df_test])

data_dict = {}

grouped_train = df_train.groupby(["original_video_id"])
grouped_test = df_test.groupby(["original_video_id"])

data_db = {}
count = 0
for name, group in grouped_train:
    
    start_frame = int(group['frame_id'].min())/30
    end_frame = int(group['frame_id'].max())/30
    time_till_now = end_frame
    label = int(group["label"].mode().iloc[0])
    #classes = num_classes + 1 (background) with last category as background
    # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
    subset = "training"
    duration = end_frame-start_frame
    file_name = name
    data_db[file_name]={
        "start":start_frame,
        "end":end_frame,
        "label":label,
        "duration":duration,
        "subset":subset,
        "time_till_now":time_till_now
    }
    if label != 0:
        count+=1
print("train number : ",count)

count = 0
for name, group in grouped_test:

    start_frame = int(group['frame_id'].min())/30
    end_frame = int(group['frame_id'].max())/30
    time_till_now = end_frame
    label = int(group["label"].mode().iloc[0])
    subset = "testing"
    duration = end_frame-start_frame
    file_name = name
    data_db[file_name]={
        "start":start_frame,
        "end":end_frame,
        "label":label,
        "duration":duration,
        "subset":subset,
        "time_till_now":time_till_now
    }
    if label != 0:
        count+=1
print("test number : ",count)


data_db_all = {}
for k,v in data_db.items():
    file_name  = k.split("_")[0]
    last_time = v["time_till_now"]

    if file_name not in data_db_all.keys():
        data_db_all[file_name] = {}
        data_db_all[file_name]["annotation"]= []
        data_db_all[file_name]["last_time"] = last_time
        if not v["label"] == 0:
            data_db_all[file_name]["annotation"].append(v)
    else:
        if last_time>data_db_all[file_name]["last_time"]:
            data_db_all[file_name]["last_time"] = last_time
        if not v["label"] == 0:
            data_db_all[file_name]["annotation"].append(v)




with open('data.json', 'w') as f:
    json.dump(data_db_all, f)