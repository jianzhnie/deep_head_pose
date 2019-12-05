#!/bin/sh

python get_faceboxes.py --root_data /media/dm/d/data/pose/gaosi/images/chinese \
                        --output_dir /media/dm/d/data/pose/gaosi/result/chinese


python get_faceboxes.py --root_data /media/dm/d/data/pose/gaosi/images/english \
                        --output_dir /media/dm/d/data/pose/gaosi/result/english


python get_faceboxes.py --root_data /media/dm/d/data/pose/gaosi/images/math \
                        --output_dir /media/dm/d/data/pose/gaosi/result/math



# ######################
# python mtcnn_headpose.py --root_data /media/dm/d/data/pose/gaosi/images/english \
#                         --output_dir /media/dm/d/data/pose/gaosi/result/english


# python mtcnn_headpose.py --root_data /media/dm/d/data/pose/gaosi/images/math \
#                         --output_dir /media/dm/d/data/pose/gaosi/result/math