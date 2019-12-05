import os, json
import sys
import cv2
import shutil
import numpy as np
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'headpose'))

import headpose_estimator


image_dir = '/media/dm/d/data/pose/XMCdata/20190605_stu'
face_info_path = '/media/dm/d/data/pose/XMCdata/20190605_stu_faceinfo.txt'
data_root = '/media/dm/d/data/pose/XMCdata'

unlabel_img_dir = '/media/dm/d/data/pose/XMCdata/unlabeled_images'
corse_labels_dir = '/media/dm/d/data/pose/XMCdata/corse_labels'


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
        image_name = image_name.replace('/data/NVR', data_root)
        bboxes_num = int(data[i+1][:-1])
        bboxes = []
        for j in range(i+2, i+2+bboxes_num):
            bbox = data[j][:-1].split(' ')
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
    mkdir(unlabel_img_dir)
    mkdir(corse_labels_dir)
    estimator = headpose_estimator.HeadPoseEstimator()
    # image_bboxes_info is a dict
    image_bboxes_info = read_bbox_from_txt(face_info_path)
    face_new_bbox = {}
    count = 0
    face_img_count = 0
    for image_name in image_bboxes_info:
        # if face_img_count > 1000:
        #     break
        if image_name.find('NVR_ch2') !=-1:
            count += 1
            print('useimage conut %d' % count)
            bboxes = image_bboxes_info[image_name]
            image = cv2.imread(image_name)
            for i, bbox in enumerate(bboxes):
                headpose = detect_headpose(image, bbox, estimator)
                headpose = headpose[0]
                yaw = headpose[0]
                pitch = headpose[1]
                roll = headpose[2]
                if within_range(yaw) and within_range(pitch) and within_range(roll):
                    face_img_count += 1
                    print("generate image count: %d" %face_img_count)
                    img_h,img_w = image.shape[:2]
                    x1,y1,w,h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    ad = 0.6
                    ## 为了方便标注，将检测到的人脸框按照一定的比例外扩
                    xmin_crop = max(int(x1 - ad * w), 0)
                    ymin_crop = max(int(y1 - ad * h), 0)
                    xmax_crop = min(int(x2 + ad * w), img_w - 1)
                    ymax_crop = min(int(y2 + ad * h), img_h - 1)
                    ## 裁剪待标注图片
                    face_crops = image[ymin_crop:ymax_crop,xmin_crop:xmax_crop]
                    face_img_name = image_name.split('/')[-2] + '_' + image_name.split('/')[-1].split('.')[0] + '_' + str(i)
                    face_img_path = os.path.join(unlabel_img_dir, face_img_name + '.jpg')
                    cv2.imwrite(face_img_path, face_crops)

                    ## 计算人脸框在新的裁剪之后的图像中的坐标
                    ## 人脸框的大小未发生变化，只是相对位置发生了变化
                    new_bbox = [0,0,0,0]
                    new_bbox[0] = x1 - xmin_crop
                    new_bbox[1] = y1 - ymin_crop
                    new_bbox[2] = new_bbox[0] + w
                    new_bbox[3] = new_bbox[1] + h 
                    face_new_bbox[face_img_name] = new_bbox

                    face_headpose = {}
                    face_headpose['yaw'] = convert_angle(headpose[0])
                    face_headpose['pitch'] = convert_angle(headpose[1])
                    face_headpose['roll'] = convert_angle(headpose[2])

                    # 将模型预测的三个角度输出到文件
                    headpose_name = 'corse_' + face_img_name + '.json'
                    headpose_file = os.path.join(corse_labels_dir, headpose_name)
                    with open(headpose_file, 'w') as f:
                        json.dump(face_headpose, f)
    
    new_bbox_file = os.path.join(data_root, 'new_bbox.json')
    with open(new_bbox_file, 'w') as f:
        json.dump(face_new_bbox,f)


if __name__ =='__main__':
    main()