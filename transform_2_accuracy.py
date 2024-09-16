import pandas as pd
import numpy as np
import sklearn
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score


gt_csv_path = "path to gt csv"
pred_csv_path = "path to predicted csv"

gt_df = pd.read_csv(gt_csv_path,usecols=["video-id","t-start","t-end","label"])
pred_df = pd.read_csv(pred_csv_path,usecols=["video-id","t-start","t-end","label","score"])
with open("path to gt json","r") as f:
    gt_json = json.load(f)


pred_df = pred_df.sort_values(by=["video-id","score"])

threshold_list = np.linspace(0.1,0.3,11)


def calculate_acc(pred_df,gt_df):
    df_gt= pd.DataFrame(columns=["video_id","gt_labels"])
    for k,v in gt_json.items():
        if "train" in k:
            continue
        dur = v["last_time"]
        video_name = k.split(".")[0]
        gt_labels = [0 for i in range(int(np.round(dur)))]
        video_id = [k for i in range(int(np.round(dur)))]
        df = pd.DataFrame(columns=["video_id","gt_labels"])
        df["video_id"] = video_id
        df["gt_labels"] = gt_labels
        subset = gt_df[gt_df["video-id"]==video_name]
        
        for index in subset.index:
            start = int(np.round(subset.loc[index]["t-start"]))
            end = int(np.round(subset.loc[index]["t-end"]))
            label = subset.loc[index]["label"]
            for i in range(start,end+1):
                df.loc[i,"gt_labels"] = label+1
        
        df_gt = pd.concat([df_gt,df],ignore_index=True)    
    df_pred = pd.DataFrame(columns=["video_id","pred_labels"])


    for k,v in gt_json.items():
        if "train" in k:
            continue
        video_name = k.split(".")[0]
        dur = v["last_time"]
        gt_labels = [0 for i in range(int(np.round(dur)))]
        video_id = [k for i in range(int(np.round(dur)))]
        df = pd.DataFrame(columns=["video_id","pred_labels"])
        df["video_id"] = video_id
        df["pred_labels"] = gt_labels
        subset = pred_df[pred_df["video-id"]==video_name]
        
        for index in subset.index:
            start = int(np.round(subset.loc[index]["t-start"]))
            end = int(np.round(subset.loc[index]["t-end"]))
            label = subset.loc[index]["label"]
            for i in range(start,end+1):
                if i>=len(df):
                    continue
                df.loc[i,"pred_labels"] = label+1
        #print(df)
        df_pred = pd.concat([df_pred,df],ignore_index=True)

    df_all =pd.concat([df_gt,df_pred["pred_labels"]],axis=1)
    acc = accuracy_score(df_all["gt_labels"].to_list(),df_all["pred_labels"].to_list())
    f1 = f1_score(df_all["gt_labels"].to_list(),df_all["pred_labels"].to_list(),average="weighted")
    recall = recall_score(df_all["gt_labels"].to_list(),df_all["pred_labels"].to_list(),average="weighted")
    precision = precision_score(df_all["gt_labels"].to_list(),df_all["pred_labels"].to_list(),average="weighted")
    jaccard = jaccard_score(df_all["gt_labels"].to_list(),df_all["pred_labels"].to_list(),average="weighted")

    df_pred.to_csv("df_pred.csv",index=False)
    df_gt.to_csv("df_gt.csv",index=False)
    print("accuracy under the threshold : ", acc,end="     ")
    print("f1 score under the threshold : ", f1,end="     ")
    print("precision score under the threshold : ", precision,end="     ")
    print("recall score under the threshold : ", recall)
    print("jaccard score under the threshold : ", jaccard)


pred_df_threshold = pred_df[pred_df["score"]>0.20]
calculate_acc(pred_df_threshold)



