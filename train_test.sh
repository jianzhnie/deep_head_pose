#!/bin/bash

### 服务器 68.58 

##  Pose_300W_LP datasets
python main.py --dataset Pose_300W_LP \
                --data_dir /data/pose/300W_LP \
                --json_path  datasets/filenamelists/300W_LP_facebbox_disgard.json \
                --filename_list  datasets/filenamelists/300W_LP_disgard.txt \
                --test_data_dir /data/pose/AFLW2000  \
                --test_json_path datasets/filenamelists/AFLW2000_facebbox_disgard.json \
                --test_filename_list datasets/filenamelists/AFLW2000_disgard.txt \
                --model resnet18 \
                --lr 1e-5 --batch_size 64 \
                --epochs 15 --gpu 4 \
                --p 100 --alpha 1 \
                --resize 256 --input_size 224 \
                --out_dir Pose_300W_LP_adrand_alpha1 \
                --tensorboard tf_logs/Pose_300W_LP_adrand_alpha1               

# AFLW2000 test
python test_hopenet_time_gpu.py --dataset AFLW2000 \
                --data_dir /data/pose/AFLW2000  \
                --filename_list datasets/filenamelists/AFLW2000_disgard.txt \
                --json_path  datasets/filenamelists/AFLW2000_facebbox_disgard.json \
                --output_dir /data/pose/XMCdata/XMC_3W_Test/JD_2W_test \
                --model resnet18 \
                --gpu 0  \
                --batch_size 1 \
                --save_viz False \
                --snapshot checkpoint/Pose_300W_LP_ad0.2_alpha1/model_best.pth

# XMCTest
python tester.py --dataset AFLW2000 \
                --data_dir  /data/pose/AFLW2000  \
                --filename_list datasets/filenamelists/AFLW2000_disgard.txt \
                --json_path  datasets/filenamelists/AFLW2000_facebbox_disgard.json \
                --output_dir /data/pose/XMCdata/XMC_3W_Test/JD_2W_test \
                --model resnet18 \
                --gpu 0  \
                --batch_size 32 \
                --snapshot checkpoint/XMC-Reg-local_headpose_regressor.pth.tar

## Tensorboard
tensorboard --logdir='./logs'
