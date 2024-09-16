import os
import random
from io import BytesIO
import torch
import numpy as np
import cv2
from slowfast.datasets.utils import pack_pathway_output
from configs.custom_config import load_config
from slowfast.utils.parser import parse_args



class VideoSet(torch.utils.data.Dataset):
    """
    transform a video into multiple sequence of frames
    一个视频就是一个数据集 
    数据集包含多个sequence of frames
    """

    def __init__(self, vid_path,cfg):
        """
        Construct the video loader for a given video.
        Args:
            cfg: configs
            vid_path: video_path
        """
        self.cfg = cfg

        self.vid_path = vid_path

        self.in_fps = cfg.DATA.IN_FPS #视频fps
        self.out_fps = cfg.DATA.OUT_FPS #输出feature的fps
        self.step_size = int(self.in_fps / self.out_fps) #输出视频的step

        #default 32
        self.seq_len = cfg.DATA.NUM_FRAMES

        if isinstance(cfg.DATA.SAMPLE_SIZE, list):
            self.sample_width, self.sample_height = cfg.DATA.SAMPLE_SIZE
        elif isinstance(cfg.DATA.SAMPLE_SIZE, int):
            self.sample_width = self.sample_height = cfg.DATA.SAMPLE_SIZE
        else:
            raise Exception(
                "Error: Frame sampling size type must be a list [Height, Width] or int"
                            )

        self.frames = self._get_frames()

    def _get_frames(self):
        '''
        将video转换成sequence of frames
        '''
        

        cap = cv2.VideoCapture(self.vid_path)

        total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = np.empty(shape=(total_count,int(self.sample_height),int(self.sample_width),3))
        print("total count: ",total_count)
        count = 0
        while(True):
            
            ret,frame = cap.read()
            if not ret:
                break
            #resize frame
            frame = cv2.resize(frame,(self.sample_height,self.sample_width),interpolation = cv2.INTER_LINEAR)

            frames[count,:,:,:] = frame

            if count%100==0:
                print("frame count:",count)
            count+=1
        #预处理全部的frames
        frames = self._pre_process_frame(frames)


        cap.release()

        return frames


    def _pre_process_frame(self, arr):
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
        arr = arr - torch.tensor(self.cfg.DATA.MEAN)
        #_C.DATA.STD = [0.225, 0.225, 0.225]
        arr = arr / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        try:
            arr = arr.permute(3, 0, 1, 2)
        except Exception as e:
            print("length of the array is not T x H x W x C ")

        return arr


    def __getitem__(self, index):
        '''
        根据index得到全部self.frames中的其中一个sequence 不需要标签 
        index被设置为seq正中间的index
        start和end分别往两边扩展

        '''

        frame_seq = torch.zeros(
            (
                3,
                self.seq_len, #default 32
                self.sample_width,
                self.sample_height
            )
        )

        #序列起始与结束index
        start_index = int(index - self.step_size*self.seq_len/2)
        end_index = int(index + self.step_size*self.seq_len/2)

        max_index = self.__len__()-1

        for new_index,old_index  in enumerate(range(start_index,end_index,self.step_size)):
            if start_index<0 or end_index>max_index:
                continue
            else:
                frame_seq[:,new_index,:,:] = self.frames[:,old_index,:,:]

        # create the pathways 即 list of slow and fast downsampling
        frame_list = pack_pathway_output(self.cfg, frame_seq)

        # frame list contains slow path frame list and fast path frame list
        return frame_list
    


    def __len__(self):

        return self.frames.shape[1]
    




if __name__ == "__main__":
    print("debuging this file ...")
    
    args = parse_args()
    cfg = load_config(args)
    
    """cfg = {
        "DATA":
        {"IN_FPS":30,
        "OUT_FPS":30,
        "SAMPLE_SIZE":224,
        "MEAN" : [0.45, 0.45, 0.45],
        "STD":[0.225, 0.225, 0.225],
        "NUM_FRAMES":8
        }

    }"""
    dataset = VideoSet("/home/pangy/disk/LUO/slowfast/videos/sample.mp4",cfg)

    temp = dataset.__getitem__(0)
    temp2 = dataset.__getitem__(100)
    print("finish debug")