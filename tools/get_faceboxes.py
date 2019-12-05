import cv2
import os
import time
import torch
import argparse
import json
import shutil
import string

from headpose import headpose_estimator

from face_detection import FaceDetector
from face_detection.config import cfg as detection_cfg

face_detector = FaceDetector(detection_cfg)

parser = argparse.ArgumentParser()
parser.add_argument('--root_data', dest='root_data', help='Path to test dataset', default='', type=str)
parser.add_argument('--output_dir', dest='output_dir', help='Path to output', default='', type=str)
args = parser.parse_args()


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def main():
    print(args)
    root_path = args.root_data
    out_root_path = args.output_dir

    estimator=headpose_estimator.HeadPoseEstimator()
    for path1 in os.listdir(root_path):
        data_dir1 = os.path.join(root_path, path1)
        out_dir1 = os.path.join(out_root_path, path1)
        isExists = os.path.exists(out_dir1)
        if not isExists:
            os.makedirs(out_dir1)
            print('path of %s is build'%(out_dir1))
        else:
            shutil.rmtree(out_dir1)
            os.makedirs(out_dir1)
            print('path of %s already exist and rebuild'%(out_dir1))
        for path2 in os.listdir(data_dir1):
            data_dir2 = os.path.join(data_dir1, path2)
            out_dir2 = os.path.join(out_dir1, path2)
            isExists = os.path.exists(out_dir2)
            if not isExists:
                os.makedirs(out_dir2)
                print('path of %s is build'%(out_dir1))
            else:
                shutil.rmtree(out_dir2)
                os.makedirs(out_dir2)
            if 'record' in data_dir2:
                continue
            headpose_dict = {}
            for filename in os.listdir(data_dir2):
                imgpath = os.path.join(data_dir2,filename)
                t1 = time.time()
                image = cv2.imread(imgpath)
                bboxes = face_detector.detect_faces(image)
                #image = face_detector.draw_bboxes(image, bboxes)
                print('faces:', len(bboxes))
                print('detect face time', time.time() - t1)
                if len(bboxes)>0:
                    headposes = estimator.estimate_headpose(image, bboxes)
                    # print("yaw :", headposes[0][0])
                    # print("pitche :", headposes[0][1])
                    # print("roll:", headposes[0][2])
                    yaw = int(headposes[0][0])
                    pitch = int(headposes[0][1])
                    roll = int(headposes[0][2])
                    headpose_dict[filename] = [yaw, pitch, roll]

            headpose_name = path2 + '_direction.json'
            headpose_file = os.path.join(out_dir2, headpose_name)
            with open(headpose_file, 'w') as f:
                json.dump(headpose_dict, f)

if __name__ == '__main__':
    main()