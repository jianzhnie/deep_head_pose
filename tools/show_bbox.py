import matplotlib.pyplot as plt
from random import random as rand
import numpy as np
import math
from math import cos, sin
import cv2


def show_boxes(im, dets, ad = 0.5, scale = 1.0, savepath=None):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    img_h,img_w = im.shape[:2]

    for i, det in enumerate(dets):
        bbox = det[:4] * scale
        x1,y1,x2,y2 = bbox
        w = x2-x1
        h = y2-y1
        xmin_crop = max(int(x1 - ad * w), 0)
        ymin_crop = max(int(y1 - ad * h), 0)
        xmax_crop = min(int(x2 + ad * w), img_w - 1)
        ymax_crop = min(int(y2 + ad * h), img_h - 1)
        
        color = (1,1,0)
        rect = plt.Rectangle((xmin_crop,ymin_crop),
                                xmax_crop - xmin_crop,
                                ymax_crop - ymin_crop,fill=False,
                                edgecolor=color, linewidth= 1.5)
        plt.gca().add_patch(rect)
    
        if dets.shape[1]== 7:
            yaw, pitch, roll= det[4:]
            plt.gca().text(xmin_crop, ymin_crop,
                            '{:s} {:.0f}, {:s} {:.0f},{:s} {:.0f}'.format('yaw', yaw, 'pitch',pitch,'roll', roll),
                            bbox=dict(facecolor=color, alpha=0.5), fontsize=5, color='white')

    if savepath != None:
        plt.savefig(savepath)
    plt.show()
    return im


def cv2_boxes(im, dets, ad = 0.6, scale = 1.0, savepath=None):
    img_h,img_w = im.shape[:2]

    for i, det in enumerate(dets):
        bbox = det[:4] * scale
        x1,y1,x2,y2 = bbox
        w = x2-x1
        h = y2-y1
        xmin_crop = max(int(x1 - ad * w), 0)
        ymin_crop = max(int(y1 - ad * h), 0)
        xmax_crop = min(int(x2 + ad * w), img_w - 1)
        ymax_crop = min(int(y2 + ad * h), img_h - 1)
        
        cv2.rectangle(im,(xmin_crop,ymin_crop),(xmax_crop,ymax_crop),(0,255,255),2)

    if savepath != None:
        cv2.imwrite(savepath, im)
    return im


def draw_axis(img, dets, tdx = None, tdy = None, size = 50):

    for i, det in enumerate(dets):
        
        bbox = det[:4]
        xmin,ymin,xmax,ymax = bbox

        yaw, pitch, roll= det[4:]

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            tdx = (xmin + xmax)/2
            tdy = (ymin + ymax)/2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy


        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),1)