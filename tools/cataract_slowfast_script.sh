#!/bin/bash

echo "start training"
python train.py ./configs/cataract_slowfast.yaml --output pretrained
echo "start testing...010"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_010.pth.tar
echo "start testing...050"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_050.pth.tar
echo "start testing...080"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_080.pth.tar
echo "start testing...090"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_090.pth.tar
echo "start testing...100"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_100.pth.tar
echo "start testing...110"
python eval.py ./configs/cataract_slowfast.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt/cataract_slowfast_pretrained/epoch_110.pth.tar
