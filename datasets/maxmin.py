"""Get the min value, max value of the roll, yaw, pitch
"""
import os
import utils
import numpy as np

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def main():
    data_dir = '/home/dm/dataset/300W_LP'
    filename_path = 'datasets/filenamelist/filenames.txt'
    filename_list = get_list_from_filenames(filename_path)
    X_train = filename_list
    y_train = filename_list

    yaw_min_max = [0, 0]
    pitch_min_max = [0, 0]
    roll_min_max = [0, 0]

    for index in range(len(y_train)):
        mat_path = os.path.join(data_dir, y_train[index] + '.mat')
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        if yaw < yaw_min_max[0]:
            yaw_min_max[0] = yaw
        elif yaw > yaw_min_max[1]:
            yaw_min_max[1] = yaw

        if pitch < pitch_min_max[0]:
            pitch_min_max[0] = pitch
        elif pitch > pitch_min_max[1]:
            pitch_min_max[1] = pitch
        
        if roll < roll_min_max[0]:
            roll_min_max[0] = roll
        elif roll > roll_min_max[1]:
            roll_min_max[1] = roll

    print('yaw', yaw_min_max)
    print('pitch', pitch_min_max)
    print('roll', roll_min_max)

if __name__ == '__main__':
    main()