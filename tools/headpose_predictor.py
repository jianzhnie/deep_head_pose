import os, json
import sys
import cv2
import shutil
import numpy as np
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'headpose'))
import headpose_estimator


from show_bbox import show_boxes,cv2_boxes
from show_pose import draw_axis


image_dir = '/media/dm/d/data/XMC/XMCtest/images'
#face_info_path = '/media/dm/d/data/pose/concentration_data/concentration_face_dets.txt'
face_info_path = '/media/dm/d/data/XMC/XMCtest/facebbox.json'
data_root = '/media/dm/d/data/XMC/XMCtest'
output_dir = '/media/dm/d/data/XMC/XMCtest/student_headpose'



def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file


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
            x,y,w,h = bbox[:4]
            xmin = int(x)
            ymin = int(y)
            xmax = xmin + int(w)
            ymax = ymin + int(h)
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
        image_bboxes_info[image_name] = bboxes
        i = i + 2 + bboxes_num
    return image_bboxes_info


def SingleBbox():
    image_bboxes_info = load_json(face_info_path)
    estimator = headpose_estimator.HeadPoseEstimator()
    count = 0
    ad = 0.2
    student_headpose = {}
    for image_name in image_bboxes_info:
        count += 1
        newbboxes = []
        print('useimage count %d' % count)
        bboxes = image_bboxes_info[image_name]
        image = cv2.imread(image_name)
        outimage = image_name.split('/')[-1]
        img_h = image.shape[0]
        img_w = image.shape[1]
        for i, bbox in enumerate(bboxes):
            bbox_proc = [[bbox[0], bbox[1], bbox[2],bbox[3]]]
            headpose = estimator.estimate_headpose(image, bbox_proc)
            headpose = headpose[0]
            yaw = int(headpose[0])
            pitch = int(headpose[1])
            roll = int(headpose[2])
            temp = [bbox[0], bbox[1], bbox[2],bbox[3], yaw, pitch, roll]
            newbboxes.append(temp)

        student_headpose[outimage] = newbboxes

    new_bbox_file = os.path.join(data_root, 'student_headpose.json')
    with open(new_bbox_file, 'w') as f:
        json.dump(student_headpose,f)


def main():
    estimator = headpose_estimator.HeadPoseEstimator()
    # image_bboxes_info is a dict
    #image_bboxes_info = read_bbox_from_txt(face_info_path)
    image_bboxes_info = load_json(face_info_path)

    count = 0
    ad = 0.2
    for image_name in image_bboxes_info:
        count += 1
        print('useimage count %d' % count)
        bboxes = image_bboxes_info[image_name]
        image = cv2.imread(image_name)
        img_h = image.shape[0]
        img_w = image.shape[1]
        outimage = image_name.split('/')[-1]
        headpose = estimator.estimate_headpose(image, bboxes)
        bboxes = np.array(bboxes)
        headposedets = np.concatenate((bboxes, headpose), axis=1)
        #print(headposedets)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i, pose in enumerate(headposedets):
            xmin,ymin,xmax,ymax = pose[:4]
            yaw, pitch, roll= pose[4:]
            ctr_x = (xmin + xmax)/2
            ctr_y = (ymin + ymax)/2
            image = draw_axis(image, yaw, pitch,roll, tdx=ctr_x,tdy=ctr_y)
        savepath = os.path.join(output_dir,outimage)
        cv2_boxes(image, headposedets,savepath=savepath)


if __name__ =='__main__':
    SingleBbox()
    main()