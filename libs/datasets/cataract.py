import os
import json
import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations


@register_dataset("cataract")
class CataractDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets train_split: [ 'training' ] val_split: [ 'validation' ]
            feat_folder,  # folder for features
            json_file,  # csv file for annotations 
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of surgical phase categories
            file_ext,  # feature file extension if any (should be npy)
            force_upsampling,  # force to upsample to max_seq_len
            backbone_type,  # feat_type slowfast
    ):
        # file path
        if backbone_type != 'tsp':
            feat_folder = os.path.join(feat_folder, split[0])
        #csv
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.backbone_type = backbone_type
        self.file_ext = file_ext

        #label and frame file
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info 从feature extract yaml文件中获取
        self.feat_stride = feat_stride
        self.num_frames = num_frames #32
        self.input_dim = input_dim #2048+256=2304
        self.default_fps = default_fps #30/15
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json(self.json_file)
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db #frame list
        self.label_dict = label_dict #label list

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Cataract',
            'tiou_thresholds': np.linspace(0.1, 0.9, 9),
            # 'tiou_thresholds': np.array([0.5, 0.75, 0.95]),
            'empty_label_ids': []
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json(self, json_file):


        #create a csv
        #frame_start,frame_end,label,training_or_testing
        # load database and select the subset
        with open(json_file) as f:
            json_data = json.load(f)
        # if label_dict is not available, matching label(str) to label id (int)
        if self.label_dict is None:
            label_dict = {
            #define surgical phase name to phase label id
                "phase_{}".format(i+1):i for i in range(self.num_classes)
            }
            
        if self.split[0] == "training":
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
            if self.default_fps is not None:
                fps = self.default_fps
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
                if self.num_classes == 1:
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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder, video_item['id'] + self.file_ext) #ext should be _32.npy
        feats = np.load(filename)

        #shape is T x 2304
        # we support both fixed length features / variable length features
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride

        # T x C -> C x T
        if isinstance(feats, torch.Tensor):
            feats = feats.transpose(0, 1)
        else:
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

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
            if self.is_training:
                feat_len = feats.shape[1]
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= feat_len:
                        # skip an action outside of the feature map
                        continue
                    # truncate an action boundary
                    valid_seg_list.append(seg.clamp(max=feat_len))
                    # some weird bug here if not converting to size 1 tensor
                    valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': video_item['id'],
                     'feats': feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': num_frames}

        # no truncation is needed
        # truncate the features during training                                                                                                      
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
