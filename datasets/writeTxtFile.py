import os
import argparse
import scipy.io as sio
import numpy as np


def validate_pose(pose):
    """
    keep pitch, yaw, roll within [-99,99]
    """
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi
    if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
        return False
    else:
        return True

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='generate txt file for datasets')
    parser.add_argument('--dataset', dest='dataset', help='dataset name to generate txt file')
    parser.add_argument('--data_dir', dest='data_dir', help='path of the dataset')
    parser.add_argument('--disgard', dest='disgard', default=1, help='0-not disgard images, 1-disgard images')
    parser.add_argument('--output_dir', dest='output_dir', help='path of the dataset')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    count = 0
    if args.dataset == '300W_LP':
        dirs_names = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']
        txtname = os.path.join(output_dir, '300W_LP_disgard.txt')
        if args.disgard == 0:
            txtname =  os.path.join(output_dir,'300W_LP.txt')
        with open(txtname, 'w') as f:
            for dir_name in dirs_names:
                path = os.path.join(data_dir, dir_name)
                files = os.listdir(path)
                files.sort()
                if args.disgard == '0':
                    for file in files:
                        [name, label] = file.split('.')
                        if label == 'jpg':
                            f.write(os.path.join(dir_name, name) + '\n')
                            count += 1
                elif args.disgard == '1':
                    for file in files:
                        [name, label] = file.split('.')
                        if label == 'mat':
                            mat = sio.loadmat(os.path.join(path, file))
                            pre_pose_params = mat['Pose_Para'][0]
                            pose = pre_pose_params[:3]
                            if validate_pose(pose):
                                f.write(os.path.join(dir_name, name) + '\n')
                                count += 1

    elif args.dataset == 'AFLW2000':
        txtname = os.path.join(output_dir, 'AFLW2000_disgard.txt')
        if args.disgard == 0:
            txtname = os.path.join(output_dir, 'AFLW2000.txt')
        with open(txtname, 'w') as f:
            files = os.listdir(data_dir)
            files.sort()
            if args.disgard == '0':
                for file in files:
                    [name, label] = file.split('.')
                    if label == 'jpg':
                        f.write(name + '\n')
                        count += 1
            elif args.disgard == '1':
                for file in files:
                    [name, label] = file.split('.')
                    if label == 'mat':
                        mat = sio.loadmat(os.path.join(data_dir, file))
                        pre_pose_params = mat['Pose_Para'][0]
                        pose = pre_pose_params[:3]
                        if validate_pose(pose):
                            f.write(name + '\n')
                            count += 1
    print('Total:', count)
    print('DONE')

if __name__ == '__main__':
    main()
