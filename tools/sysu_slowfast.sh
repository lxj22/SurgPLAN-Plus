#!/bin/bash
echo "start training"
python train.py ./configs/sysu_slowfast.yaml --output pretrained
echo "start testing...010"
python eval.py ./configs/sysu_slowfast.yaml /data/disk/LUO/test_only/TriDet/ckpt/sysu_slowfast_pretrained/epoch_010.pth.tar
echo "start testing...050"
python eval.py ./configs/sysu_slowfast.yaml /data/disk/LUO/test_only/TriDet/ckpt/sysu_slowfast_pretrained/epoch_050.pth.tar
