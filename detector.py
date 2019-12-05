import cv2
import json
import os
import argparse
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from face_detection import FaceDetector
from face_detection.config import cfg as detection_cfg
from datasets import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', help='dataset name', default='', type=str)
parser.add_argument('--root_dir', dest='root_dir', help='Path to test dataset', default='', type=str)
parser.add_argument('--filename_list', dest='filename_list', help='filename list of images', default='', type=str)
args = parser.parse_args()


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def show_Pose_300w_LP():
    face_detector = FaceDetector(detection_cfg)
    root_dir = args.root_dir
    filenamelist = get_list_from_filenames(args.filename_list)
    for image_name in filenamelist:
        image_path = os.path.join(root_dir, image_name +".jpg")
        mat_path = os.path.join(root_dir, image_name + '.mat')
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        bboxes = np.array([[x_min,y_min, x_max, y_max, 1.0]])
        image = cv2.imread(image_path)
        image = face_detector.draw_bboxes(image, bboxes)
        cv2.imshow('image', image)
        k = cv2.waitKey(500)


def show_AFLW2000():
    face_detector = FaceDetector(detection_cfg)
    root_dir = args.root_dir
    filenamelist = get_list_from_filenames(args.filename_list)
    for image_name in filenamelist:
        image_path = os.path.join(root_dir, image_name +".jpg")
        mat_path = os.path.join(root_dir, image_name + '.mat')
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        bboxes = np.array([[x_min,y_min, x_max, y_max, 1.0]])
        image = cv2.imread(image_path)
        image = face_detector.draw_bboxes(image, bboxes)
        cv2.imshow('image', image)
        k = cv2.waitKey(500)
    return

 
def main():
    dataset = args.dataset
    root_dir = args.root_dir
    filenamelist = get_list_from_filenames(args.filename_list)
    face_detector = FaceDetector(detection_cfg)
    count = 0
    det_count =0
    faceboxes = {}
    if dataset == 'BIWI':
        img_ext='_rgb.png'
    else:
        img_ext = '.jpg'
    for image_name in filenamelist:
        image_path = os.path.join(root_dir, image_name + img_ext)
        image = cv2.imread(image_path)
        bboxes = face_detector.detect_faces(image)
        count += 1
        if len(bboxes) > 0:
            det_count +=1
            print("images, %s, detected face %s" %(count, det_count))
            x_min, y_min, x_max, y_max = bboxes[0][:4]
            xmin = int(x_min)
            ymin = int(y_min)
            xmax = int(x_max)
            ymax = int(y_max)
            bbox = [xmin, ymin, xmax, ymax]
        faceboxes[image_name] = bbox
        # if count >10:
        #     break
    headpose_file = os.path.join('datasets', 'filenamelists', dataset + '_facebbox_disgard.json')
    with open(headpose_file, 'w') as f:
        json.dump(faceboxes, f,  indent=4)
    return

if __name__== '__main__':
    #show_AFLW2000()
    #show_Pose_300w_LP()
    main()