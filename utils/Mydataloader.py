import tensorflow as tf

import random
from PIL import Image
import cv2
import numpy as np
from utils.util import *
import math
import matplotlib.pyplot as plt
import os
from utils.target_utils import target_encoder,target_decoder

#解决中文路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

class RandomTarget_dataset(tf.keras.utils.Sequence):
    def __init__(self,root: str,
                 batch_size: int,
                 bg_root,
                 bg_r=bg_r,
                 bg_w=bg_w,
                 box_r=box_r,
                 img_size=(22, 28),
                 grid=(grid_r,grid_w),
                 img_num=None,
                 Chinese_path=False,
                 valid=False):
        self.paths = self.get_paths_and_labels(root, class_num=3)
        random.shuffle(self.paths)
        if valid:
            self.paths = self.paths[:len(self.paths)//5]
        self.batch_size = batch_size
        self.bg_r = bg_r
        self.bg_w = bg_w
        self.bg_paths = [f'{bg_root}/{p}' for p in os.listdir(bg_root)]
        self.img_num=img_num
        self.Chinese_path=Chinese_path
        self.box_r = box_r
        self.img_size = img_size
        self.grid_r, self.grid_w = grid
        self.target_d = round(self.box_r*(2**0.5))#目标之间的最近距离
        self.encoder = target_encoder(input_size=(bg_r,bg_w),grid=grid)#更改target的grid数目
    def get_paths_and_labels(self,root, class_num)->list:
        #针对猪肺默认双层目录读取
        img_paths = []
        if class_num == 3:
            for cls in os.listdir(root):
                path_1 = f"{root}\\{cls}"
                for s_cls in os.listdir(path_1):
                    path_2 = f"{path_1}\\{s_cls}"
                    img_paths.append(path_2)
            return img_paths

    def creat_random_data(self,path,bg_img_path):

        bg_size = (self.bg_w, self.bg_r)

        if self.Chinese_path:
            bg_img = cv_imread(bg_img_path)
        else:
            bg_img = cv2.imread(bg_img_path)
        bg_img = cv2.resize(bg_img, bg_size)

        if self.img_num==None:
            img_num = random.randint(2, 4)
        else:
            img_num=self.img_num

        targets = []  # 图片中心的偏移
        if self.Chinese_path:
            img = cv_imread(path)
        else:
            img = cv2.imread(path)
        #去除边框      我直接用无框，不需要去除
        # img = remove_frame(img)

        l=3#边界距离
        img0 = np.zeros((self.bg_r + self.box_r, self.bg_w + self.box_r, 3), dtype=np.uint8)
        # 创建一张黑色画布,尺寸要比目标尺寸大，最后截取中间部分，这样可以包含边界不完整情况
        for i in range(img_num):
            x, y = random.randint(l, self.bg_w-1-l), random.randint(l, self.bg_r-1-l)
            t =0
            while judge_point(targets, x, y,self.target_d) != True:
                x, y = random.randint(l, self.bg_w-1-l), random.randint(l, self.bg_r-1-l)
                t+=1
                if t>=50:
                    break
            if t>=50:
                break
            size = random.randint(self.img_size[0], self.img_size[1]) // 2 * 2 + 1
            img_rotate,angle = random_rotate(img)
            if abs(angle%90) > 15:
                size = int(size*1.2)
            img_rotate = cv2.resize(img_rotate, (size, size))
            #判断旋转角度，如果角度大，就适当放大图片的大小
            _x = x+self.box_r//2-(size-1)//2
            x_ = _x + size
            _y = y+self.box_r//2-(size-1)//2
            y_ = _y + size
            img0[_y:y_,_x:x_,:] = img_rotate[:, :, :]
            #用旋转后的图片覆盖ground truth
            targets.append([x, y, size, angle])
            #将中心点和大小即旋转角度加入target

        #截取画面的中间部分，这样可以包含边界不完整情况
        img0 =  img0[self.box_r//2:self.box_r//2+self.bg_r,self.box_r//2:self.box_r//2+self.bg_w]

        mask = get_mask(img0)
        #获得一个图片部分为0，剩余为1的mask
        bg_img = cv2.bitwise_and(bg_img, bg_img, mask=mask)
        #按像素与，mask部分保留，剩下部分变0
        img1 = cv2.add(bg_img, img0)
        #把背景和图片的按照像素加和即可
        # label = encode(target_centers,self.box_r)
        label = self.encoder.encode(targets)
        """
        opencv 为bgr要转rgb
        """
        img1 = img1[:,:,::-1]
        return np.array(img1,dtype=np.float32),np.array(label,dtype=np.float32)

    def on_epoch_end(self):
        random.shuffle(self.paths)
    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)
    def __getitem__(self, item):
        batch = self.paths[item * self.batch_size:(item + 1) * self.batch_size]
        imgs = []
        labels = []
        for path in batch:
            bg_img_path = self.bg_paths[random.randint(0,len(self.bg_paths)-1)]
            img,label=self.creat_random_data(path,bg_img_path)
            imgs.append(img)
            labels.append(label)

        return np.array(imgs,dtype=np.float32),np.array(labels,dtype=np.float32)

class Fast_dataset(tf.keras.utils.Sequence):
    def __init__(self,imgs_path:str,labels_path,batch_size,input_size=(96,128)):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.imgs,self.labels = self.get_paths_and_labels(self.imgs_path,self.labels_path)
        self.input_r,self.input_w = input_size
    def get_paths_and_labels(self,imgs_path,labels_path):
        imgs = os.listdir(imgs_path)
        labels0 = np.load(labels_path)
        labels =[]
        for path in imgs:
            idx = int(path[:-4])
            labels.append(labels0[idx-1])
        return imgs,labels
    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)
    def __getitem__(self, item):
        batch_img_path = self.imgs[item * self.batch_size:(item + 1) * self.batch_size]
        batch_labels = self.labels[item * self.batch_size:(item + 1) * self.batch_size]
        imgs = []
        for path in batch_img_path:
            img = cv2.imread(f'{self.imgs_path}/{path}')
            img = cv2.resize(img,(self.input_w,self.input_r))
            img = img[:,:,::-1]
            imgs.append(img)
        return np.array(imgs, dtype=np.float32), np.array(batch_labels, dtype=np.float32)
if __name__=='__main__':
    batch_size =1
    img_num=3
    # dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
    #                       batch_size=batch_size,bg_r=96,bg_w=128,
    #                       bg_root=r'C:\Project\python\dataset\background',
    #                       img_num=img_num,
    #                       Chinese_path=True)
    dataset = Fast_dataset(
        imgs_path='../dataset/test/images',
        labels_path='../dataset/test/labels.npy',
        batch_size=batch_size
    )


    for (img,label) in dataset:
        print(img.shape)
        img1 = np.array(img[0],dtype=np.uint8)
        result_show(img1,label[0],target_decoder())
