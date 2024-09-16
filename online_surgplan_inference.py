#imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score


# Modified to process a list of videos
"""Extract features for videos using pre-trained networks"""
from feature_extract.configs.custom_config import load_config
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args
import torch
import os
import time
from tqdm import tqdm
import av
from moviepy.video.io.VideoFileClip import VideoFileClip

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from feature_extract.models import build_model
from feature_extract.datasets.extract_dataset import VideoSet
import copy
logger = logging.get_logger(__name__)
# python imports
import glob
from pprint import pprint

# torch imports
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_tridet_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

#imports
import json
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn import functional as F

from libs.datasets.datasets import register_dataset
from libs.datasets.data_utils import truncate_feats
from libs.utils import remove_duplicate_annotations

import random
from io import BytesIO
import cv2
from slowfast.datasets.utils import pack_pathway_output
from feature_extract.configs.custom_config import load_config
from slowfast.utils.parser import parse_args
from matplotlib import pyplot as plt
from collections import deque
import argparse
import sys


from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict
import datetime


#Feature extraction configs
def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["./feature_extract/slowfast/feature_extract/configs/SLOWFAST_8x8_R50_1031.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--f",
        help="for jupyternotebook to run",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


#Feature extraction image process

def pre_process_frame(arr):
        """
        Pre process an array
        Args:
            arr (ndarray): an array of frames of shape T x H x W x C 
        Returns:
            arr (tensor): a normalized torch tensor of shape C x T x H x W 
        """
        arr = torch.from_numpy(arr).float()
        # Normalize the values
        arr = arr / 255.0
        #DATA.MEAN = [0.45, 0.45, 0.45]
        arr = arr - torch.tensor([0.45, 0.45, 0.45])
        #_C.DATA.STD = [0.225, 0.225, 0.225]
        arr = arr / torch.tensor([0.225, 0.225, 0.225])

        # T H W C -> C T H W.
        try:
            arr = arr.permute(3, 0, 1, 2)
        except Exception as e:
            print("length of the array is not T x H x W x C ")

        return arr


def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds

#feature extraction slow fast inference
@torch.no_grad()
def perform_inference(inputs, model, cfg):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    feat_arr = None

    # Transfer the data to the current GPU device.
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(device=cfg.USED_GPU,non_blocking=True)
    else:
        inputs = inputs.cuda(device=cfg.USED_GPU,non_blocking=True)

    # Perform the forward pass.
    preds, feat = model(inputs)
    # Gather all the predictions across all the devices to perform ensemble.
    if cfg.NUM_GPUS > 1:
        preds, feat = du.all_gather([preds, feat])

    feat = feat.cpu().numpy()

    if feat_arr is None:
        feat_arr = feat
    else:
        feat_arr = np.concatenate((feat_arr, feat), axis=0)

    return feat_arr#video feature extraction stream output
def test(model,cfg,inputs):
    """
    Perform multi-view testing/feature extraction on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)


    feat_arr = perform_inference(inputs, model, cfg)
    

    return feat_arr

def _load_json(label_dict=None,split=['testing'],num_classes=19,json_file="data.json",default_fps=30):

        #create a csv
        #frame_start,frame_end,label,training_or_testing
        # load database and select the subset
        with open(json_file) as f:
            json_data = json.load(f)
        # if label_dict is not available, matching label(str) to label id (int)
        if label_dict is None:
            label_dict = {
            #define surgical phase name to phase label id
                "phase_{}".format(i+1):i for i in range(num_classes)
            }
            
        if split[0] == "training":
            split_name = "train"
        else:
            split_name = "test"
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_data.items():
            # key is the video id, v is the segement information
            # get fps if available

            if split_name not in value["annotation"][0]["subset"]:
                continue
            if default_fps is not None:
                fps = default_fps
            else:
                #fps=30
                fps = 1

            duration = []
            num_phase = len(value["annotation"])
            segments = np.zeros([num_phase, 2], dtype=np.float32)
            labels = np.zeros([num_phase, ], dtype=np.int64)
            for idx,phase in enumerate(value["annotation"]):
                duration.append(phase["time_till_now"])
                segments[idx][0] = phase["start"]
                segments[idx][1] = phase["end"]
                if num_classes == 1:
                    labels[idx] = 0
                else:
                    labels[idx] = phase["label"]
            last_time = value["last_time"]
            dict_db += ({'id': key.split(".")[0],
                         'fps': fps,
                         'duration': last_time,
                         'segments': segments,
                         'labels': labels
                         },)

        return dict_db, label_dict

def getitem(data_list,features, idx,num_frames,max_seq_len=1024,feat_stride=1,downsample_rate=1,force_upsampling=False,mirror=True):
    # directly return a (truncated) data point (so it is very fast!)
    # auto batching will be disabled in the subsequent dataloader
    # instead the model will need to decide how to batch / preporcess the data
    video_item = data_list[idx]

    # load features
    feats = features

    #shape is T x 2304
    # we support both fixed length features / variable length features
    if feat_stride > 0 and (not force_upsampling):
        # var length features
        feat_stride, num_frames = feat_stride, num_frames
        # only apply down sampling here
        if downsample_rate > 1:
            feats = feats[::downsample_rate, :]
            feat_stride = feat_stride * downsample_rate

    # T x C -> C x T
    if isinstance(feats, torch.Tensor):
        feats = feats.transpose(0, 1)
    else:
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

    # convert time stamp (in second) into temporal feature grids
    # ok to have small negative values here
    if video_item['segments'] is not None:
        segments = torch.from_numpy(
            #(video_item['segments'] * video_item['fps'] - 0.5 * num_frames) / (feat_stride)
            video_item['segments']
        )
        labels = torch.from_numpy(video_item['labels'])
        # for activity net, we have a few videos with a bunch of missing frames
        # here is a quick fix for training
        segments, labels = None, None

    # return a data dict
    data_dict = {'video_id': video_item['id'],
                    'feats': feats,  # C x T
                    'segments': segments,  # N x 2
                    'labels': labels,  # N
                    'fps': 30,
                    'duration': num_frames,
                    'feat_stride': 30,
                    'feat_num_frames': 32}


    return data_dict


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def valid_one_epoch(
        dict_db,
        model,
        curr_epoch=0,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
        # forward the model (wo. grad)
    with torch.no_grad():
        output = model(dict_db)

        # upack the results into ANet format
        num_vids = len(output)
        for vid_idx in range(num_vids):
            if output[vid_idx]['segments'].shape[0] > 0:
                results['video-id'].extend(
                    [output[vid_idx]['video_id']] *
                    output[vid_idx]['segments'].shape[0]
                )
                results['t-start'].append(output[vid_idx]['segments'][:, 0])
                results['t-end'].append(output[vid_idx]['segments'][:, 1])
                results['label'].append(output[vid_idx]['labels'])
                results['score'].append(output[vid_idx]['scores'])


    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    return results


def load_gt_seg_from_json(json_file, split=None, label='label', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)

    vids, starts, stops, labels = [], [], [], []
    if split == "training":
        split_name = "train"
    else:
        split_name = "test"
    for k, v in json_db.items():

        for segments in v["annotation"]:
            if split_name not in segments["subset"]:
                continue
            ants = segments
            vids.append(k.split(".")[0])
            starts.append(ants["start"])
            stops.append(ants["end"])
            labels.append(ants["label"])


    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id': vids,
        't-start': starts,
        't-end': stops,
        'label': labels
    })

    return gt_base



def to_df(preds):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id': preds['video-id'],
                't-start': preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })



        return preds

def calculate_acc(pred_df,gt_df):
    with open("data.json","r") as f:
        gt_json = json.load(f)
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


    return df_pred


def get_middle_label(pred_df,mid_index,threshold=0.22):
    df = pred_df[pred_df["score"]>threshold]
    label = 0
    score = threshold
    s = mid_index[0]
    e = mid_index[1]
    #filter df
    if len(df) == 0:
        for i in range(len(pred_df)):
            if pred_df.iloc[i]["t-start"]<=e and pred_df.iloc[i]["t-end"]>=s:
                label = 0
                score = pred_df.iloc[i]["score"]
                break
    for i in range(len(df)):
        if df.iloc[i]["t-start"]<=e and df.iloc[i]["t-end"]>=s:
            label = df.iloc[i]["label"]
            score = df.iloc[i]["score"]
            break
    return label,score


def mirror_feature(inputs,reversed_inputs,filled_inputs,num_fill):
    feat = inputs["feats"]
    feat_flipped = torch.from_numpy(reversed_inputs)
    feat_flipped = feat_flipped.transpose(0, 1)
    if num_fill>0:
        feat_fill = np.repeat(filled_inputs,num_fill,axis=0)
        
        feat_fill = torch.from_numpy(feat_fill)
        feat_fill = feat_fill.transpose(0, 1)
        feat_mirrored = torch.concat((feat, feat_fill,feat_flipped), dim=1)
        inputs["feats"] = feat_mirrored
        inputs['duration'] = inputs['duration']*2+num_fill
        ratio = inputs["duration"]//512+1
        inputs["feats"] = feat_mirrored[:,::ratio]
        inputs['duration'] = len(inputs["feats"])
    else:
        feat_mirrored = torch.concat((feat,feat_flipped), dim=1)
        inputs["feats"] = feat_mirrored
        inputs['duration'] = inputs['duration']*2
        ratio = inputs["duration"]//512+1
        inputs["feats"] = feat_mirrored[:,::ratio]
        inputs['duration'] = len(inputs["feats"])
    return inputs


def fill_feature(inputs,filled_inputs,num_fill):
    feat = inputs["feats"]
    feat_fill = np.repeat(filled_inputs,num_fill,axis=0)
    feat_fill = torch.from_numpy(feat_fill)
    feat_fill = feat_fill.transpose(0, 1)
    feat_filled = torch.concat((feat, feat_fill), dim=1)
    inputs["feats"] = feat_filled
    inputs['duration'] = inputs['duration']+num_fill
    ratio = inputs["duration"]//512+1
    inputs["feats"] = feat_filled[:,::ratio]
    inputs['duration'] = len(inputs["feats"])
    return inputs


#Feature extraction configs
args = parse_args()
cfg = load_config(args)
cfg.USED_GPU = 4 #make sure it is same as device
vid_path = "path to video"



tridet_cfg = load_tridet_config("/configs/cataract.yaml")
ckpt_file = 'model_ckpt'
topk=-1
tridet_cfg['model']['test_cfg']['max_seg_num'] = topk
print_freq = 10
save_only = False

# fix the random seeds (this will fix everything)
_ = fix_random_seed(0, include_cuda=True)




device = "cuda:0"
extract_model = build_model(cfg).to(device)
extract_checkpoint = torch.load("slowfast_model.pyth",map_location=device)
print("Loading from Feature Extraction Model ...")
extract_model.load_state_dict(extract_checkpoint['model_state'])
del extract_checkpoint
model = make_meta_arch(tridet_cfg['model_name'], **tridet_cfg['model'])
model = nn.DataParallel(model, device_ids=tridet_cfg['devices'])
print("=> loading checkpoint '{}'".format(ckpt_file))
checkpoint = torch.load(
    ckpt_file,
    map_location=lambda storage, loc: storage.cuda(tridet_cfg['devices'][0])
)
print("loading finish")
# load ema model instead
print("Loading from EMA model ...")
model.load_state_dict(checkpoint['state_dict_ema'])
del checkpoint
print("loading finish")
gt = pd.read_csv("df_gt.csv")
dir_name = "./cataract_test_video"


#debug only

def main(vid,log_folder,threshold=0.1,num_fill=32):    
    print("Eval video: ",vid," ...")
    vid_path = os.path.join(dir_name,vid)
    device = "cuda"
    cap = cv2.VideoCapture(vid_path)
    sample_height = 256
    sample_width = 256
    count = 0
    seq_length = 32
    queue = deque(maxlen=64)
    ret = True
    frame_wise_feature = None
    idx = int(vid_path.split("/")[-1].split(".")[0].split("test")[-1])-1 #test01 -> 0
    dict_db,label_dict = _load_json() #dict_db contains segments and labels
    current_db = dict_db[idx] #current test file json
    count_2 = 0 #count after queue is full
    # load tridet model
    # not ideal for multi GPU training, ok for now

    # load ckpt, reset epoch / best rmse


    pred_list = []
    score_list = []
    start = datetime.datetime.now()
    while ret:
        #print("count: ",count)
        
        ret,frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        frame = cv2.resize(frame,(sample_height,sample_width),interpolation = cv2.INTER_LINEAR)
        queue.append(frame)
        if count<64:
            count+=1
            continue
        else:
            frames = np.stack(queue, axis=0)
            frames = frames[::2,:,:,:]
            reversed_frames = np.flip(frames, axis=0)
            reversed_frames = reversed_frames.copy()
            filled_frames = np.expand_dims(frame, axis=0)
            filled_frames = np.repeat(filled_frames, 32, axis=0)
            #print(frames.shape)
            #print(reversed_frames.shape)
            #print(filled_frames.shape)
            frames = pre_process_frame(frames)
            reversed_frames = pre_process_frame(reversed_frames)
            filled_frames = pre_process_frame(filled_frames)
            
            frame_list = pack_pathway_output(cfg, frames) #得到一个list,list[0]为(3,8,256,256),[1]为[3,32,256,256]分别表示slow和fast两条线
            frame_list[0] = frame_list[0].unsqueeze(0) #增加一个batch维度
            frame_list[1] = frame_list[1].unsqueeze(0)

            reversed_frame_list = pack_pathway_output(cfg, reversed_frames)
            reversed_frame_list[0] = reversed_frame_list[0].unsqueeze(0) #增加一个batch维度
            reversed_frame_list[1] = reversed_frame_list[1].unsqueeze(0)     

            filled_frame_list = pack_pathway_output(cfg, filled_frames)
            filled_frame_list[0] = filled_frame_list[0].unsqueeze(0) #增加一个batch维度
            filled_frame_list[1] = filled_frame_list[1].unsqueeze(0)   


            if count_2%30 == 0: #每30表示的是每30个frame，即每一秒取一次feature
                cur_second = count//30
                if frame_wise_feature is None:
                    frame_wise_feature = test(extract_model,cfg,frame_list) #得到每32个frame为单位的一个feature，size大小为(1,2304)
                    reversed_frame_wise_feature = test(extract_model,cfg,reversed_frame_list)
                    filled_frame_wise_feature = test(extract_model,cfg,filled_frame_list)
                    #print("frame_wise_feture",frame_wise_feture.shape)
                    #print(test(cfg,frame_list))
                else:
                    #print("frame_wise_feture ",frame_wise_feture.shape)
                    cur_feature = test(extract_model,cfg,frame_list)
                    reversed_cur_feature = test(extract_model,cfg,reversed_frame_list)
                    #print("cur_feature ",cur_feature.shape)
                    frame_wise_feature = np.concatenate((frame_wise_feature,cur_feature),axis=0)
                    reversed_frame_wise_feature =  np.concatenate((reversed_cur_feature,reversed_frame_wise_feature),axis=0)

                    filled_frame_wise_feature = test(extract_model,cfg,filled_frame_list)
                    #print(reversed_frame_wise_feature.shape)
                    #print(filled_frame_wise_feature.shape)
                    #print(frame_wise_feture.shape)
                    #将frame_wise_feature 在dim 0 上concate起来，放入surgplan的eval方程
                
                
                #transfer input frame_wise_features into dictionary
                inputs = getitem(data_list=dict_db,features=frame_wise_feature, idx=idx,num_frames=len(frame_wise_feature))
                #inference
                mirror_features = mirror_feature(inputs,reversed_frame_wise_feature,filled_frame_wise_feature,num_fill)
                results = valid_one_epoch([mirror_features],model=model)
                df = to_df(results)
                if mirror_features["feats"].size(1) >= 1024:
                    pred_label,score = get_middle_label(df,[512-num_fill//2,512+num_fill//2],threshold)
                else:
                    pred_label,score = get_middle_label(df,[cur_second+num_fill-2,cur_second+num_fill],threshold=threshold)

                pred_list.append(pred_label)
                score_list.append(score)
            count+=1
            count_2 +=1 
    end = datetime.datetime.now()
    cap.release()
    print("total time: ", (end-start).seconds)
    acc = accuracy_score(gt[gt["video_id"]==vid]["gt_labels"][-len(pred_list):],pred_list[:])
    print("accuracy for vid: ",vid," is : ",acc)
    x = gt[gt["video_id"]==vid][-len(pred_list):]
    x["pred_labels"] = pred_list
    x["pred_scores"] = score_list

    csv_file = "./"+log_folder+"/result_real_time_"+vid+".csv"
    x.to_csv(csv_file)


cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_result"
os.mkdir(cur_time)
for i in os.listdir("/data/disk/LUO/cataract_test_video"):
    main(i,cur_time,threshold=0.18,num_fill=24)


