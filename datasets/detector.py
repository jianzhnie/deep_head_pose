import os
import cv2
from os import listdir
from os.path import isfile, join
from mtcnn.mtcnn import MTCNN
import numpy as np
import shutil


detector = MTCNN()

root_dir = '/media/dm/d/data/pose/XMCdata'
img_dir = '/media/dm/d/data/pose/XMCdata/20190605_stu'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('path of %s is build'%(path))
    else:
        shutil.rmtree(path)
        os.makedirs(path)
        print('path of %s already exist and rebuild'%(path)) 

def main():
    facebox = dict()
    for dirs in os.listdir(img_dir):
        data_dir = os.path.join(img_dir,dirs)
        imglist = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and join(data_dir, f).endswith('.jpg')]
        count = 0
        for imagename in imglist:
            if imagename.find('ch2') !=-1:
                img_path = os.path.join(data_dir, imagename)
                img = cv2.imread(img_path)
                img_h = img.shape[0]
                img_w = img.shape[1]
                detected = detector.detect_faces(img)
                print(detected)
                if len(detected) > 0:
                    for i_d, d in enumerate(detected):
                        if d['confidence'] > 0.5:
                            x1,y1,w,h = d['box']
                            x2 = x1 + w
                            y2 = y1 + h
                            ad = 0.6
                            xmin_crop = max(int(x1 - ad * w), 0)
                            ymin_crop = max(int(y1 - ad * h), 0)
                            xmax_crop = min(int(x2 + ad * w), img_w - 1)
                            ymax_crop = min(int(y2 + ad * h), img_h - 1)
                            cv2.rectangle(img, (xmin_crop, ymin_crop), (xmax_crop, ymax_crop), (0, 255, 0), 2)  
                    cv2.imshow('test', img)
                    k = cv2.waitKey(10)
                else:
                    count +=1
                    #print('no face detected')
                    print(count)


if __name__ =='__main__':
    main()