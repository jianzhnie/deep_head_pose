import sys, os, argparse
import math, time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from headpose.hopenet import Hopenet
from datasets import MyDatasets, utils
from headpose.mobilenet_v2 import MobileNetV2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='Directory path for output_dir data.',
          default='', type=str)
    parser.add_argument('--json_path', dest='json_path', help='Directory path for image bboxes info.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshots.',
          default='', type=str)
    parser.add_argument('--model', dest='model', help='model to use', default='resnet50', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu
    output_dir = args.output_dir
    # backbone structure
    print ("===> creating Hopenet model by '{}'".format(args.model))
    if args.model == 'resnet50':
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 120)
    elif args.model == 'resnet18':
        model = Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 120)
    elif args.model == 'mobilenet_v2':
        model = MobileNetV2(num_bins=120, width_mult=1.0)

    if os.path.isfile(args.snapshot):
        print("===> loading model weight from '{}'".format(args.snapshot))
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['state_dict'])
    else:
        print("===> no found at '{}'".format(args.snapshot))
        print("loading model weight from models/resnet50_epoch_4.pkl ")
        model.load_state_dict(torch.load("models/resnet50_epoch_4.pkl"))

    print('===> Loading data.')
    
    resize = 128
    input_size = 112
    transformations = transforms.Compose([
                        transforms.Resize((resize, resize)),
                        transforms.CenterCrop(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = MyDatasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = MyDatasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = MyDatasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = MyDatasets.AFLW2000_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = MyDatasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = MyDatasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = MyDatasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = MyDatasets.AFW(args.data_dir, args.filename_list, transformations)
    elif args.dataset =='XMCTest':
        pose_dataset = MyDatasets.XMCTestData(args.data_dir, args.data_dir, args.json_path, args.filename_list, transformations)
    elif args.dataset == 'XMC':
        pose_dataset = MyDatasets.XMCData(args.data_dir, args.filename_list, transformations)
    else:
        print ('Error: not a valid dataset name')
        sys.exit()

    ## dataloader     
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4)
    model.cuda(gpu)
    print ('===> Ready to test network.')
    # Test the Model
    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()  
    total = 0

    idx_tensor = [idx for idx in range(120)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    
    result = []
    T = 0
    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        end = time.time()
        images = Variable(images).cuda(gpu)
        cont_labels = Variable(cont_labels)
        total += cont_labels.size(0)

        #ctr_x, ctr_y = ctr
        label_yaw = cont_labels[:,0].float()
        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

        yaw, pitch, roll = model(images)
        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        # yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        # pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        # roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = F.softmax(yaw.data, dim=1)
        pitch_predicted = F.softmax(pitch.data, dim=1)
        roll_predicted = F.softmax(roll.data, dim=1)
        
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 1 - 60
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 1 - 60
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 1 - 60

        # Mean absolute error
        yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
        pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
        roll_error += torch.sum(torch.abs(roll_predicted - label_roll))
        mae = (yaw_error +pitch_error+roll_error)/3
        
        t = time.time() - end
        T += t

        result.append([yaw_error.item()/total,pitch_error.item()/total,roll_error.item()/total, mae.item()/total])

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            if args.dataset == 'BIWI':
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '_rgb.png'))
            else:
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
            if args.batch_size == 1:
                #utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
                cv2_img_test = cv2_img
                #cv2_img_test = utils.draw_axis(cv2_img_test, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
                cv2_img = utils.draw_axis(cv2_img, label_yaw,label_pitch, label_roll, size=100)
                # show the result
                #cv2.imshow("gt",cv2_img)
                #cv2.waitKey(500)
                #cv2.imwrite(os.path.join(output_dir, name + '_test'+ '.jpg'), cv2_img_test)
                cv2.imwrite(os.path.join(output_dir, name + '_gt'+ '.jpg'), cv2_img)

        # writer.add_scalar('Test/yaw', yaw_error.item()/total, i)
        # writer.add_scalar('Test/pitch', pitch_error.item()/total, i)
        # writer.add_scalar('Test/roll', roll_error.item()/total, i)
        # writer.add_scalar('Test/mae', mae.item()/total, i)
        
        print('Test error in degrees of the model on the ' + str(total) +
        ' test images. Time: %.4f, Yaw: %.4f, Pitch: %.4f, Roll: %.4f MAE: %4f' % (t, yaw_error / total, pitch_error / total, roll_error / total, mae/total))
    
    avg = T / i
    print("avg time %f" %avg)
    # name=['yaw','pitch','roll','mae']
    # test=pd.DataFrame(columns=name,data=result)
    # test.to_csv('result.csv')

if __name__ == '__main__':
    main()