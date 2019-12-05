#!/bin/sh
python main.py --dataset XMCTest \
            --data_dir /data/pose/XMCdata/XMC_3W/train --filename_list  datasets/filenamelists/XMC/XMC_filenamelist.txt \
            --json_path  datasets/filenamelists/XMC/face_info_bbox.json \
            --test_data_dir /data/pose/XMCdata/XMC_3W/train --test_filename_list datasets/filenamelists/XMC/cls2/val.txt \
            --test_json_path datasets/filenamelists/XMC/face_info_bbox.json \
            --model resnet18 \
            --resize 128 --input_size 112 \
            --lr 1e-5 --batch_size 32 \
            --epochs 20 --gpu 1 \
            --p 100 --alpha 1 \
            --out_dir resnet18_lapha1_lr1e-5_bs32_input112_cls2_half_all_data \
            --tensorboard tf_logs/resnet18_lapha1_lr1e-5_bs32_input112_cls2_half_all_data \
            --half_train \
            --eps 1e-7


python main_xmc.py --dataset XMCTest \
            --data_dir /data/pose/lexue/lexue1000 --filename_list  datasets/filenamelists/lexue/lexue_filenamelist.txt \
            --json_path  datasets/filenamelists/lexue/face_info_bbox.json \
            --test_data_dir /data/pose/lexue/lexue1000 --test_filename_list datasets/filenamelists/lexue/lexue_filenamelist.txt \
            --test_json_path datasets/filenamelists/lexue/face_info_bbox.json \
            --model resnet18 \
            --fintune  --fintune_model models/XMC2-Reg_face_direction_regression.pth.tar \
            --resize 128 --input_size 112 \
            --lr 1e-5 --batch_size 32 \
            --epochs 20 --gpu 1 \
            --p 100 --alpha 1 \
            --out_dir resnet18_lapha1_lr1e-5_bs32_input112_half \
            --tensorboard tf_logs/resnet18_lapha1_lr1e-5_bs32_input112_half \
            --half_train \
            --eps 1e-7