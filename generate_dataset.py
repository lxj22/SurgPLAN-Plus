import pandas as pd
import json
import os
import numpy as np
import argparse



def main(train_csv_path,test_csv_path,json_path):
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

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
        count+=1
    print("train segments number : ",count)

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
        count+=1
    print("test segments number : ",count)


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



    with open(json_path, 'w') as f:
        json.dump(data_db_all, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='')
    parser.add_argument('--test_path',  default='')
    parser.add_argument('--json_path',  default='')

    args = parser.parse_args()

    print("generating json file : ",args.json_path)
    main(args.train_path,args.test_path,args.json_path)
    print("finish")

    