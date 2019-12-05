import cv2
import os
import shutil
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from face_detection import FaceDetector
from face_detection.config import cfg as detection_cfg
from headpose import headpose_estimator


face_detector = FaceDetector(detection_cfg)

root_dir = '/home/dm/jianzh/pose/tuoyun/newimages'
output_root = '/home/dm/jianzh/pose/tuoyun/headpose/cube'
ad = 0.2

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('path of %s is build'%(path))
    else:
        shutil.rmtree(path)
        os.makedirs(path)
        print('path of %s already exist and rebuild'%(path)) 

def main():
    estimator=headpose_estimator.HeadPoseEstimator()
    for dir in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, dir)
        output_dir = os.path.join(output_root, dir)
        mkdir(output_dir)
        for path in os.listdir(img_dir):
            img_path = os.path.join(img_dir,path)
            image = cv2.imread(img_path)
            img_h = image.shape[0]
            img_w = image.shape[1]
            detected = detector.detect_faces(image)
            if len(detected) > 0:
                for i_d, d in enumerate(detected):
                    if d['confidence'] > 0.90:
                        x1,y1,w,h = d['box']
                        x2 = x1 + w
                        y2 = y1 + h
                        xmin_crop = max(int(x1 - ad * w), 0)
                        ymin_crop = max(int(y1 - ad * h), 0)
                        xmax_crop = min(int(x2 + ad * w), img_w - 1)
                        ymax_crop = min(int(y2 + ad * h), img_h - 1)
                        ctr_x = (xmin_crop + xmax_crop)/2
                        ctr_y = (ymin_crop + ymax_crop)/2
                        face_crops = image[ymin_crop:ymax_crop,xmin_crop:xmax_crop]
                        bbox = [[xmin_crop, ymin_crop, xmax_crop, ymax_crop]]
                        headpose = estimator.estimate_headpose(image, bbox)
                        headpose = headpose[0]
                        yaw = headpose[0]
                        pitch = headpose[1]
                        roll = headpose[2]
                        #estimator.draw_axis(image,yaw, pitch, roll, tdx = ctr_x, tdy=ctr_y)
                        estimator.plot_pose_cube(image,yaw, pitch, roll, tdx = ctr_x, tdy=ctr_y)
                        output_img = os.path.join(output_dir, path)
                        cv2.imwrite(output_img, image)

if __name__ == '__main__':
    main()
