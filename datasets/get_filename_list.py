import os
from os import listdir
from os.path import isfile, join
import numpy as np

root = '/media/dm/d/data/pose/BIWI'
XMC_data = '/media/dm/d/data/pose/XMCdata/XMC_3W/JD_2W'


data_path = ['AFW','AFW_Flip','HELEN','HELEN_Flip','IBUG','IBUG_Flip','LFPW','LFPW_Flip']
data_path = ['0'+str(x) for x in range(1,10)]
data_path.extend([str(x) for x in range(10,25)])


def validate_pose(pose):
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi
    if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
        return False
    else:
        return True


def validate_biwi_pose(pose_path):
    pose_annot = open(pose_path, 'r')
    R = []
    for line in pose_annot:
        line = line.strip('\n').split(' ')
        l = []
        if line[0] != '':
            for nb in line:
                if nb == '':
                    continue
                l.append(float(nb))
            R.append(l)

    R = np.array(R)
    T = R[3,:]
    R = R[:3,:]
    pose_annot.close()

    R = np.transpose(R)

    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

    if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
        return False
    else:
        return True


def get_single_file_list(mypath_obj):
    onlyfiles_mat_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.json')]
    onlyfiles_jpg_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.jpg')]
    return onlyfiles_mat_temp, onlyfiles_jpg_temp 


def get_300WLP_list():
    filenamelist = []
    for path in data_path:
        print(path)
        mypath = os.path.join(root, path)
        print('Read %s file list' % mypath)
        onlyfiles_mat_temp, onlyfiles_jpg_temp = get_single_file_list(mypath)
        print(len(onlyfiles_mat_temp))
        print(len(onlyfiles_jpg_temp))
        onlyfiles_mat_temp.sort()
        onlyfiles_jpg_temp.sort()
        for idx in range(len(onlyfiles_mat_temp)):
            img_name = onlyfiles_jpg_temp[idx]
            txt_name = onlyfiles_jpg_temp[idx]
            img_name = img_name.split('.')[0]
            txt_name = txt_name.split('.')[0]
            if img_name == txt_name:
                filepath = os.path.join(path, img_name)
                filenamelist.append(filepath)
            else:
                print('Mismatched !!!')
    return filenamelist


def get_BIWI_list():
    filenamelist = []
    disgard_filenem_list = []
    for path in data_path:
        mypath = os.path.join(root, path)
        print('Read %s file list' % mypath)
        onlyfiles_mat_temp = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.txt')]
        onlyfiles_jpg_temp = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.png')]
        print(len(onlyfiles_mat_temp))
        print(len(onlyfiles_jpg_temp))
        onlyfiles_mat_temp.sort()
        onlyfiles_jpg_temp.sort()
        for idx in range(len(onlyfiles_mat_temp)):
            img_name = onlyfiles_jpg_temp[idx]
            txt_name = onlyfiles_mat_temp[idx]
            imgname = img_name.split('.')[0]
            txtname = txt_name.split('.')[0]
            imname = imgname[:-4]
            txtname = txtname[:-5]
            if imname == txtname:
                filepath = os.path.join(path, imname)
                filenamelist.append(filepath)
            else:
                print('Mismatched !!!')
            posefile = os.path.join(root, path, txt_name)
            if validate_biwi_pose(posefile):
                filepath = os.path.join(path, txtname)
                disgard_filenem_list.append(filepath)

    return filenamelist, disgard_filenem_list


def get_XMC_list():
    filenamelist = []
    onlyfiles_mat_temp, onlyfiles_jpg_temp = get_single_file_list(XMC_data)
    print(len(onlyfiles_mat_temp))
    print(len(onlyfiles_jpg_temp))
    for idx in range(len(onlyfiles_mat_temp)):
        img_name = onlyfiles_jpg_temp[idx]
        txt_name = onlyfiles_jpg_temp[idx]
        img_name = img_name.split('.')[0]
        txt_name = txt_name.split('.')[0]
        if img_name == txt_name:
            filenamelist.append(img_name)
        else:
            print('Mismatched !!!')
    return filenamelist


def main():
    #filenamelist = get_300WLP_list()
    #filenamelist,disgard_filenem_list = get_BIWI_list()
    filenamelist = get_XMC_list()
    with open('filenamelists/JD_2W.txt','w+') as f:
        for line in filenamelist:
            f.write(line + '\n')
        print('Done !!!')
    # with open('filenamelists/BIWI_disgard.txt','w+') as f:
    #     for line in disgard_filenem_list:
    #         f.write(line + '\n')


if __name__== '__main__':
    main()