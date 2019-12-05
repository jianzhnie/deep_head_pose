import os, json
import sys
import cv2
import shutil
import numpy as np


image_dir = '/media/dm/d/data/pose/XMCdata/20190605_stu'
face_info_path = '/media/dm/d/data/pose/XMCdata/20190605_stu_faceinfo.txt'
data_root = '/media/dm/d/data/pose/XMCdata'



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


def main():
    # image_bboxes_info is a dict
    image_bboxes_info = read_bbox_from_txt(face_info_path)
    face_new_bbox = {}
    count = 0
    face_img_count = 0
    for image_name in image_bboxes_info:
        if image_name.find('NVR_ch2') !=-1:
            count += 1
            print('useimage conut %d' % count)
            bboxes = image_bboxes_info[image_name]
            for i, bbox in enumerate(bboxes):
                face_img_count += 1
                print("generate image count: %d" %face_img_count)
                img_h,img_w = image.shape[:2]
                x1,y1,w,h = bbox
                x2 = x1 + w
                y2 = y1 + h
                ad = 0.6
                xmin_crop = max(int(x1 - ad * w), 0)
                ymin_crop = max(int(y1 - ad * h), 0)
                xmax_crop = min(int(x2 + ad * w), img_w - 1)
                ymax_crop = min(int(y2 + ad * h), img_h - 1)

                ## 计算人脸框在新的裁剪之后的图像中的坐标
                ## 人脸框的大小未发生变化，只是相对位置发生了变化
                face_img_name = image_name.split('/')[-2] + '_' + image_name.split('/')[-1].split('.')[0] + '_' + str(i)
                new_bbox = [0,0,0,0]
                new_bbox[0] = x1 - xmin_crop
                new_bbox[1] = y1 - ymin_crop
                new_bbox[2] = new_bbox[0] + w
                new_bbox[3] = new_bbox[1] + h 
                face_new_bbox[face_img_name] = new_bbox

    new_bbox_file = os.path.join(data_root, '20190605_stu_faceinfo_crop_jianzh.json')
    with open(new_bbox_file, 'w') as f:
        json.dump(face_new_bbox,f)


if __name__ =='__main__':
    main()