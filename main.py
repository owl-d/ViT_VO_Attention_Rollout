from torch.utils.data import DataLoader
import torch

import cv2 as cv
import numpy as np

from dataset_generator_KITTI_Stack import KITTI_dataset_generator
from dataloader_KITTI_Stack import KITTI_dataset



file_make = False

if file_make == True : #True이면 h5py 파일 생성
    kitti_dataset_generator = KITTI_dataset_generator(dataset_save_path='KITTI_stack_dataset_info.hdf5',
                                dataset_path='/media/doyu/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color')

#데이터셋
kitti_dataset_training = KITTI_dataset(dataset_path='KITTI_stack_dataset_info.hdf5', mode='training')
kitti_dataset_validation = KITTI_dataset(dataset_path='KITTI_stack_dataset_info.hdf5', mode='validation')
kitti_dataset_test = KITTI_dataset(dataset_path='KITTI_stack_dataset_info.hdf5', mode='test')

#데이터로더
training_dataloader = DataLoader(dataset=kitti_dataset_training, batch_size=3, shuffle=False, num_workers=0)
validation_dataloader = DataLoader(dataset=kitti_dataset_validation, batch_size=1, shuffle=False,num_workers=0)
test_dataloader = DataLoader(dataset=kitti_dataset_test, batch_size=1, shuffle=False, num_workers=0)


# if __name__ == '__main__':
for i_batch, (image, pose_list) in enumerate(training_dataloader):

    #dataloader가 주는 데이터의 shape
    print("\nimage shape : ", image.size())         # batch_size X 376 X 1241 X channel(6)
    print("pose diff list shape : ", pose_list.size())   # batch_size X 6
    print(" ")
    
    #dataloader의 tensor 데이터를 gpu에 올리기
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #cuda machine 환경이면 "cuda:0"
    print(device)
    image.to(device)
    pose_list.to(device)

    batch = len(pose_list)

    # 6DoF 차이값 출력
    for i in range(batch):
        print('\nbatch', i_batch, "-", i, end=' ')
        print("pose(6DoF) diff : \n x : {}, y : {}, z : {} \n roll : {}, pitch : {}, yaw : {}"
                .format(pose_list[i][0], pose_list[i][1], pose_list[i][2], pose_list[i][3], pose_list[i][4], pose_list[i][5]))

    # 해당 이미지 출력 (한 배치의 이미지들을 하나의 이미지로)
    for i in range(batch):
        img = image[i]
        img = img.numpy()
        img = cv.resize(img, dsize=(620, 188), interpolation=cv.INTER_LINEAR)

        if i==0:
            addh = img
        else:
            addh = np.hstack((addh, img))
        
    # cv.imshow('image', addh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 처음 샘플에 대해서만 하기
    if i_batch == 0:
        break