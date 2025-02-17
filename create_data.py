from utils.Mydataloader import RandomTarget_dataset
import numpy as np
import cv2
import os
from utils.util import *
from utils.target_utils import *
from parameters import *

def create_data(root,create_num,img_num,overwrite,img_path,bg_path,bg_r=bg_r,bg_w=bg_w,img_size=(22,28),grid=(grid_r,grid_w)):
    batch_size = 1
    if os.path.exists(f'{root}/images') is not True:
        os.makedirs(f'{root}/images')
    if overwrite:
        t=0
        labels=[]
    else:

        t, labels = check(root)
    while t<create_num:
        dataset = RandomTarget_dataset(root=img_path,
                                       batch_size=batch_size,
                                       bg_r=bg_r,
                                       bg_w=bg_w,
                                       bg_root=bg_path,
                                       img_num=img_num,
                                       img_size=img_size,
                                       grid=grid,
                                       Chinese_path=True,
                                       valid=True)
        # save_data(dataset,f'{root}/images',train_num)
        for (img, label) in dataset:
            img1 = np.array(img[0], dtype=np.uint8)
            labels.append(label[0])

            """
            opencv 为bgr要转rgb
            """
            img1 = img1[:, :, ::-1]
            t += 1
            print(t)
            cv2.imwrite(f'{root}/images/{t}.jpg', img1)
            if t >= create_num:
                break
    labels = np.array(labels)
    np.save(f'{root}/labels.npy',labels)
def check(root):
    t = len(os.listdir(f'{root}/images'))
    labels = np.load(f'{root}/labels.npy')
    labels = labels.tolist()
    print(t,len(labels))
    assert t==len(labels)
    return t,labels

#旧版编码转新版编码
# def labels_transformation(label_path):
#     labels =np.load(label_path)
#     new_labels = []
#     encoder = target_encoder()
#     for label in labels:
#         coords = decode(label)
#         size = np.zeros((coords.shape[0],1))
#         coords = np.concatenate((coords,size),axis=1)
#         new_label = encoder.encode(coords)
#         new_labels.append(new_label)
#         # print(new_label)
#     np.save(label_path, new_labels)
if __name__ == '__main__':

    train_num=10000
    valid_num=1000
    test_num=30
    labels =[]
    train_root = './dataset/train'
    valid_root = './dataset/valid'
    test_root = './dataset/test'

    img_path  = r'D:\Python Project\Project1\venv\dataset'
    bg_path = r'D:\Python Project\object-locating\background'

    overwrite = True
    img_num = 5
    create_data(root=train_root, create_num=train_num, img_num=img_num, overwrite=overwrite,img_path=img_path,bg_path=bg_path,
                bg_r=bg_r,bg_w=bg_w,img_size=img_size,grid=(grid_r,grid_w))
    create_data(root=valid_root,create_num=valid_num,img_num=img_num,overwrite=overwrite,img_path=img_path,bg_path=bg_path,
                bg_r=bg_r,bg_w=bg_w,img_size=img_size,grid=(grid_r,grid_w))
    # create_data(root=test_root,create_num=test_num,img_num=img_num,overwrite=overwrite,img_path=img_path,bg_path=bg_path,
    #             bg_r=bg_r,bg_w=bg_w,img_size=img_size,grid=(grid_r,grid_w))
