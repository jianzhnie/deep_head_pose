import sys, os, argparse, time
import warnings
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.autograd import Variable

import torch.utils.data
import torch.utils.data.distributed
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms

from headpose.hopenet import Hopenet
from headpose.mobilenet_v2 import MobileNetV2
from datasets import MyDatasets
from torch.utils.tensorboard import SummaryWriter
from datasets import utils


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    ## training & optimizer
    parser.add_argument('--epochs', dest='epochs', help='Maximum number of training epochs.',
            default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')        
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
            default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')   
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    ## distribution
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    # model seting
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
            default=1, type=float)       
    parser.add_argument('--model', dest='model', help='model to use', default='resnet18', type=str)
    parser.add_argument('--half_train', dest='half_train', action='store_true',
                        help='training with half precision')
    parser.add_argument('--input_size', dest='input_size', help='Input size for CNN Network.',
                        default=112, type=int)
    parser.add_argument('--resize', dest='resize', help='Resize the Input image for CNN Network.',
                        default=112, type=int)                        
    parser.add_argument('--eps', dest='eps', help='eps for adam optimizer',
                        default=1e-8, type=float)

    ## dataset
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
            default='', type=str)
    parser.add_argument('--json_path', dest='json_path', help='Directory path for image bboxes info.',
          default='', type=str)           
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
            default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--out_dir', default=None, type=str, help='output path of the checkpoint file') 
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Path to test dataset', default='', type=str)
    parser.add_argument('--test_filename_list', dest='test_filename_list', help='Path to test file containing \
                            relative paths for test example.', default='', type=str)
    parser.add_argument('--test_json_path', dest='test_json_path', help='Directory path for image bboxes info.',
          default='', type=str)  
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    args = parser.parse_args()
    return args


minmum_loss = np.inf

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_ignored_params_mobilenetv2(model):
    # Generator function that yields ignored params.
    b = model.features[0]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            print(module_name)


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    # b = [model.layer3, model.layer4] 
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def loss_mean(loss):
    loss_sum = 0
    for l in loss:
        loss_sum += l
    return loss_sum / len(loss)


def main():
    args = parse_args()
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    outdir = os.path.join('checkpoint', args.out_dir)
    isExists = os.path.exists(outdir)
    if not isExists:
        os.makedirs(outdir)
        print('===> Path of %s is build'%(outdir))  

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global minmum_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # backbone structure
    print ("===> Creating Hopenet model by '{}'".format(args.model))
    if args.model == 'resnet50':
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    elif args.model == 'resnet18':
        model = Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    elif args.model == 'resnet152':
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
    elif args.model == 'mobilenet_v2':
        model = MobileNetV2(num_bins=66, width_mult=1.0)

    if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(args.workers / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if args.half_train:
        print('===> Training with Half Precision')
        model = model.half()
      
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    reg_criterion = nn.MSELoss().cuda(args.gpu)
    if args.half_train:
        criterion = criterion.half()
        reg_criterion = reg_criterion.half()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    if args.model == 'mobilenet_v2':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, eps=args.eps)
    else:
        # optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
        #                     {'params': get_non_ignored_params(model), 'lr': args.lr},
        #                     {'params': get_fc_params(model), 'lr': args.lr * 5}],
        #                     lr = args.lr)
        optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': args.lr},
                                        {'params': get_non_ignored_params(model), 'lr': args.lr},
                                        {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                        lr = args.lr, eps=args.eps)

    # optimizer = Adam16([{'params': get_ignored_params(model), 'lr': 0},
    #                                 {'params': get_non_ignored_params(model), 'lr': args.lr},
    #                                 {'params': get_fc_params(model), 'lr': args.lr * 5}],
    #                                 lr = args.lr )

    # optionally resume from a checkpoint
    if args.resume == '':
        print("===> Trained without resume, load imagenet pretrained model: '{}'".format(args.model))
        if args.model == 'resnet50':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
        elif args.model == 'resnet18':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        elif args.model == 'mobilenet_v2':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'))

    else:
        if os.path.isfile(args.resume):
            print("===> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            minmum_loss = checkpoint['minmum_loss']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     minmum_loss = minmum_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    print ('===> Loading data.')
    resize = args.resize
    input_size = args.input_size
    transformations = transforms.Compose([
                        transforms.Resize(resize),
                        #transforms.Resize((resize,resize)),
                        transforms.RandomCrop(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])

    test_transformations = transforms.Compose([
                        transforms.Resize(resize),
                        #transforms.Resize((resize,resize)),
                        transforms.CenterCrop(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])
    if args.dataset == 'Pose_300W_LP':
        pose_dataset = MyDatasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = MyDatasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, args.json_path, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = MyDatasets.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = MyDatasets.AFLW2000(args.data_dir, args.filename_list, args.json_path, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = MyDatasets.BIWIDet(args.data_dir, args.filename_list, args.json_path, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = MyDatasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = MyDatasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = MyDatasets.AFW(args.data_dir, args.filename_list, transformations)
    elif args.dataset =='XMCTest':
        pose_dataset = MyDatasets.XMCTestData(args.data_dir, args.data_dir, args.json_path, args.filename_list, transformations)
    else:
        print ('Error: not a valid dataset name')
        sys.exit()

    # data loader
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(pose_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True,
                                                num_workers=args.workers, 
                                                pin_memory=True, sampler=train_sampler)

    test_pose_dataset = MyDatasets.AFLW2000(args.test_data_dir, 
                                            args.test_filename_list, 
                                            args.test_json_path, 
                                            transformations)

    # test_pose_dataset = MyDatasets.XMCTestData(args.test_data_dir, 
    #                                             args.test_data_dir,
    #                                             args.test_json_path,
    #                                             args.test_filename_list, 
    #                                             test_transformations)

    test_loader = torch.utils.data.DataLoader(dataset=test_pose_dataset,
                                                batch_size=args.batch_size, 
                                                shuffle=False,
                                                num_workers=args.workers, 
                                                pin_memory=True)

    print ('===> Ready to train network.')
    if args.tensorboard is not None:
        print("===> Use tensorboard")
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/"  + "train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + "val")
    
    if args.evaluate:
        test_error = validate(test_loader, model, criterion, reg_criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        print()
        print("base_lr:{} \t adjust_lr:{} \t batch-size:{}".format(args.lr, lr, args.batch_size))

        # train for one epoch
        train_loss = train(train_loader, model, criterion, reg_criterion, optimizer, epoch, args)
        train_loss_yaw, train_loss_pitch, train_loss_roll, train_loss_mae = train_loss
        # evaluate on validation set
        test_error = validate(test_loader, model, criterion, reg_criterion,args)
        test_yaw_error, test_pitch_error, test_roll_error, test_mae_error = test_error
        
        if args.tensorboard is not None:
            train_writer.add_scalar("yaw_loss", train_loss_yaw, epoch)
            train_writer.add_scalar("pitch_loss", train_loss_pitch, epoch)
            train_writer.add_scalar("roll_loss", train_loss_roll, epoch)
            train_writer.add_scalar("mean_loss", train_loss_mae, epoch)

            val_writer.add_scalar("yaw_error", test_yaw_error, epoch)
            val_writer.add_scalar("pitch_error", test_pitch_error, epoch)
            val_writer.add_scalar("roll_error", test_roll_error, epoch)
            val_writer.add_scalar("mean_error", test_mae_error, epoch)

        # remember best acc@1 and save checkpoint
        is_best = test_mae_error < minmum_loss
        minmum_loss = min(test_mae_error, minmum_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'minmum_loss': minmum_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = os.path.join('checkpoint', args.out_dir, args.model + '_' + str(args.alpha) + '_' + str(epoch) + '.pth'), out_dir = args.out_dir)

def train(train_loader, model, criterion, reg_criterion, optimizer, epoch, args):
    # LOG
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    yaw_loss = AverageMeter('Loss-Yaw', ':.4e')
    pitch_loss = AverageMeter('Loss-Pitch', ':.4e')
    roll_loss = AverageMeter('Loss-Roll', ':.4e')
    mae =  AverageMeter('MAE', ':.4e')

    progress = ProgressMeter(len(train_loader), batch_time, data_time, yaw_loss, 
                                pitch_loss, roll_loss, mae, prefix="Epoch: [{}]".format(epoch))
    
    # Regression loss coefficient
    alpha = args.alpha
    if args.half_train:
        alpha = torch.Tensor([[alpha]]).cuda(args.gpu).half()
        alpha = alpha.item()

    softmax = nn.Softmax(dim=1).cuda(args.gpu)
    # form -99 to 102, every 3 degree generate a bin
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(args.gpu)

    if args.half_train:
        idx_tensor = idx_tensor.half()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, labels, cont_labels, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            cont_labels = cont_labels.cuda(args.gpu, non_blocking=True)
        if args.half_train:
            #print("====> Images.cuda().half()")
            images = images.half()
            # labels = labels.half()
            cont_labels = cont_labels.half()
        
        # Binned labels
        label_yaw = Variable(labels[:,0]).cuda(args.gpu, non_blocking=True)
        label_pitch = Variable(labels[:,1]).cuda(args.gpu, non_blocking=True)
        label_roll = Variable(labels[:,2]).cuda(args.gpu, non_blocking=True)

        # Continuous labels
        label_yaw_cont = Variable(cont_labels[:,0]).cuda(args.gpu, non_blocking=True)
        label_pitch_cont = Variable(cont_labels[:,1]).cuda(args.gpu, non_blocking=True)
        label_roll_cont = Variable(cont_labels[:,2]).cuda(args.gpu, non_blocking=True)

        # Forward pass
        # yaw : batch_size * 67
        yaw, pitch, roll = model(images)
        # Cross entropy loss
        loss_yaw = criterion(yaw, label_yaw)
        loss_pitch = criterion(pitch, label_pitch)
        loss_roll = criterion(roll, label_roll)

        # MSE loss
        yaw_predicted = softmax(yaw)
        pitch_predicted = softmax(pitch)
        roll_predicted = softmax(roll)

        ## is an expection
        ## 计算期望

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

        loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
        loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
        loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

        # Total loss
        loss_yaw += alpha * loss_reg_yaw
        loss_pitch += alpha * loss_reg_pitch
        loss_roll += alpha * loss_reg_roll
        loss_mae = (loss_yaw +loss_pitch+loss_roll)/3
        
        # measure accuracy and record loss
        yaw_loss.update(loss_yaw.item(), images.size(0))
        pitch_loss.update(loss_pitch.item(), images.size(0))
        roll_loss.update(loss_roll.item(),images.size(0))
        mae.update(loss_mae.item(), images.size(0))

        # compute gradient and do SGD step
        loss_seq = [loss_yaw, loss_pitch, loss_roll]
        grad_seq = [torch.ones(1).cuda(args.gpu) for _ in range(len(loss_seq))]
        if args.half_train:
            grad_seq = [torch.ones(1).cuda(args.gpu).half() for _ in range(len(loss_seq))]

        optimizer.zero_grad()
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.print(i)

    return  yaw_loss.avg, pitch_loss.avg, roll_loss.avg, mae.avg

def validate(val_loader, model, criterion,reg_criterion, args):
    # LOG
    batch_time = AverageMeter('Time', ':6.3f')
    test_yaw_error = AverageMeter('Error-Yaw', ':.4e')
    test_pitch_error = AverageMeter('Error-Pitch', ':.4e')
    test_roll_error = AverageMeter('Eroor-Roll', ':.4e')
    test_mae =  AverageMeter('MAE', ':.4e')

    progress = ProgressMeter(len(val_loader), batch_time, test_yaw_error, 
                                test_pitch_error, test_roll_error, test_mae, prefix="Test:")
    
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(args.gpu)

    if args.half_train:
        idx_tensor = idx_tensor.half()

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, labels, cont_labels, name) in enumerate(val_loader):
            # measure data loading time
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
                cont_labels = cont_labels.cuda(args.gpu, non_blocking=True)
            
            if args.half_train:
                #print("====> Images.cuda().half()")
                images = images.half()
                cont_labels = cont_labels.half()

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(args.gpu, non_blocking=True)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(args.gpu, non_blocking=True)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(args.gpu, non_blocking=True)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = F.softmax(yaw.data, dim=1)
            pitch_predicted = F.softmax(pitch.data, dim=1)
            roll_predicted = F.softmax(roll.data, dim=1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1)* 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1)* 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1)* 3 - 99

            # Mean absolute error
            yaw_error = torch.sum(torch.abs(yaw_predicted - label_yaw_cont))
            pitch_error = torch.sum(torch.abs(pitch_predicted - label_pitch_cont))
            roll_error = torch.sum(torch.abs(roll_predicted - label_roll_cont))

            mae = (yaw_error + pitch_error + roll_error) / 3

            # measure accuracy and record loss
            batch_size = images.size(0) 
            test_yaw_error.update(yaw_error.item()/batch_size , images.size(0))
            test_pitch_error.update(pitch_error.item()/batch_size, images.size(0))
            test_roll_error.update(roll_error.item()/batch_size, images.size(0))
            test_mae.update(mae.item()/batch_size, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' ***** test_yaw:{test_yaw_error.avg:.3f} test_pitch:{test_pitch_error.avg:.3f}  test_roll:{test_roll_error.avg:.3f} test_mae:{test_mae.avg:.3f}'
                    .format(test_yaw_error=test_yaw_error, test_pitch_error=test_pitch_error, test_roll_error= test_roll_error, test_mae=test_mae))

    return test_yaw_error.avg, test_pitch_error.avg, test_roll_error.avg, test_mae.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filename, out_dir):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('checkpoint', out_dir, 'model_best.pth'))


if __name__=='__main__':
    main()
