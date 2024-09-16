#!/bin/bash

echo "start training"
python train.py /home/pangy/disk/LUO/test_only/TriDet/configs/cataract_slowfast_no_finetune.yaml --output pretrained
echo "start testing...010"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_010.pth.tar
echo "start testing...050"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_050.pth.tar
echo "start testing...080"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_080.pth.tar
echo "start testing...090"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_090.pth.tar
echo "start testing...100"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_100.pth.tar
echo "start testing...110"
python eval.py ./configs/cataract_slowfast_no_finetune.yaml /home/pangy/disk/LUO/test_only/TriDet/ckpt_none_finetune/cataract_slowfast_no_finetune_pretrained/epoch_110.pth.tar
