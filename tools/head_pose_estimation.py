"""
这个脚本文件读取image和对应的检测器生成的 bbox 文件，
然后利用 headpose estimator 预测 人脸朝向的三个角度，并将结果保存为 csv 格式
"""

import os, json
import sys
import cv2
import shutil
import numpy as np
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'headpose'))

import headpose_estimator


image_dir = '/media/dm/d/data/pose/concentration_data/images'
face_info_path = '/media/dm/d/data/pose/concentration_data/concentration_face_dets.txt'
data_root = '/media/dm/d/data/pose/concentration_data'
output_dir = '/media/dm/d/data/pose/concentration_data/headpose'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('path of %s is build'%(path))
    else:
        shutil.rmtree(path)
        os.makedirs(path)
        print('path of %s already exist and rebuild'%(path)) 


def read_bbox_from_txt(txt_file_path):
    """
    read bbox from widerface file
    input : widerface filepath, a txt file
    output: a dict,with keys is the image name
    """
    image_bboxes_info = {}
    with open(txt_file_path, 'r') as f:
        data = f.readlines()
    i = 0
    while i < len(data):
        image_name = data[i][:-1]
        image_name = image_name.replace('/media/dm/d/data/DATAset/concentration_zhuojunwei', data_root)
        bboxes_num = int(data[i+1][:-1])
        bboxes = []
        for j in range(i+2, i+2+bboxes_num):
            bbox = data[j][:-1].split(' ')
            bbox = bbox[:4]
            bbox = [int(x) for x in bbox]
            bboxes.append(bbox)
        image_bboxes_info[image_name] = bboxes
        i = i + 2 + bboxes_num
    return image_bboxes_info


def detect_headpose(image, bbox, headpose_estimator):
    """
    input: image
           bbox: detected bbox from detector
           headpose_estimator: a class function
    """
    bbox_proc = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]]
    headpose = headpose_estimator.estimate_headpose(image, bbox_proc)

    return headpose

def convert_angle(angle):
    """
    将角度的预测值由小数转为整数, 每5度一个间隔
    """
    quotient = angle // 5
    remainder = angle % 5
    if remainder < 2.5:
        new_angle = quotient * 5
    else:
        new_angle = (quotient + 1) * 5
    
    return int(new_angle)

def within_range(angle):
    """
    标注工具的度量范围：[-60, 60]
    生成的模板图片的范围：[-55, 55]
    """
    if angle >= -57.5 and angle < 57.5:
        return True
    else:
        return False

def main():
    estimator = headpose_estimator.HeadPoseEstimator()
    # image_bboxes_info is a dict
    image_bboxes_info = read_bbox_from_txt(face_info_path)
    head_pose_bbox = {}
    count = 0
    ad = 0.2
    for image_name in image_bboxes_info:
        count += 1
        face_new_bbox = []
        print('useimage conut %d' % count)
        bboxes = image_bboxes_info[image_name]
        image = cv2.imread(image_name)
        img_h = image.shape[0]
        img_w = image.shape[1]
        outimage = image_name.split('/')[-1]
        for i, bbox in enumerate(bboxes):
            headpose = detect_headpose(image, bbox, estimator)
            headpose = headpose[0]
            yaw = int(headpose[0])
            pitch = int( headpose[1])
            roll = int(headpose[2])
            x1,y1,w,h = bbox
            x2 = x1 + w
            y2 = y1 + h
            xmin_crop = max(int(x1 - ad * w), 0)
            ymin_crop = max(int(y1 - ad * h), 0)
            xmax_crop = min(int(x2 + ad * w), img_w - 1)
            ymax_crop = min(int(y2 + ad * h), img_h - 1)
            ctr_x = (xmin_crop + xmax_crop)/2
            ctr_y = (ymin_crop + ymax_crop)/2
            face_new_bbox.append([x1,y1, w, h, yaw, pitch, roll])
            cv2.rectangle(image,(xmin_crop,ymin_crop),(xmax_crop,ymax_crop),(255,0,0),2)
            cv2.putText(image, "y:%d,p:%d,r:%d" %(yaw, pitch,roll), (xmin_crop, ymin_crop - 1), fontFace=1, fontScale=1,color=(0,0,255), thickness=1)
            image = estimator.draw_axis(image,yaw, pitch, roll, tdx = ctr_x, tdy=ctr_y)

        output_img = os.path.join(output_dir,outimage)
        print(output_img)
        # cv2.imshow('test',image)
        # k = cv2.waitKey(1000)
        cv2.imwrite(output_img, image)
        head_pose_bbox[image_name] = face_new_bbox
    head_pose_file = os.path.join(data_root, 'concentration_data.json')
    with open(head_pose_file, 'w') as f:
        json.dump(head_pose_bbox,f)

if __name__ =='__main__':
    main()


# import os
# import json

# def load_json(json_path):
#     with open(json_path, 'r') as f:
#         json_file = json.load(f)
#     return json_file

# data_root = '/media/dm/d/data/pose/concentration_data'

# head_pose = load_json('/media/dm/d/data/pose/concentration_data/concentration_data.json')
# new_head_pose = {}
# for image_name in head_pose:
#     bbox = head_pose[image_name]
#     new_image_name = image_name.replace(data_root, '/media/dm/d/data/DATAset/concentration_zhuojunwei')
#     new_head_pose[new_image_name] = bbox

# head_pose_file = os.path.join(data_root, 'concentration_data2.json')
# with open(head_pose_file, 'w') as f:
#     json.dump(new_head_pose,f)