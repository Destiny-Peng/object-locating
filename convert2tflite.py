import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow import keras
from keras.models import load_model
from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
import json
import time
import os
from utils.target_utils import *

def result_show(img,label,decoder,pred,input_size):
    pred_pos = decoder.decode(pred)
    for i in range(grid_r):
        for j in range(grid_w):
            y, x = 32 * i, 32 * j
            img = cv2.rectangle(img, (x, y), (x + 32, y + 32), color=(0, 0, 255))
    img_pos = decoder.decode(label)
    for point in img_pos:
        x, y, angle = point
        cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 255, 0))
    for point in pred_pos:
        x, y, angle = point
        cv2.circle(img, (int(x), int(y)), radius=2, color=(255, 0, 0))
        cv2.rectangle(img, (int(x)-size//2*2+1, int(y)-size//2*2+1), (int(x)+size//2*2+1, int(y)+size//2*2+1), color=(255, 0, 0))
    # img = cv2.resize(img,(input_size[1],input_size[0]))
    plt.figure()
    plt.imshow(img)
    plt.show()
def tflite_pre(modelpath,dataset_root,batch_size=1,input_size:tuple = (96,128)):
    test_root = dataset_root
    interpreter = tf.lite.Interpreter(model_path=modelpath)
    input_index = interpreter.get_input_details()
    output_index = interpreter.get_output_details()
    valid_dataset =  Fast_dataset(
        imgs_path=f'{test_root}/images',
        labels_path=f'{test_root}/labels.npy',
        batch_size=batch_size,
        input_size=input_size
    )
    for (img,label) in valid_dataset:
        # img1 = np.array(img[0], dtype=np.uint8)
        # interpreter.resize_tensor_input(input_index[0]['index'], (1, input_size[0], input_size[1], 3))
        # interpreter.resize_tensor_input(output_index[0]['index'], (1, 36))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index[0]['index'], np.array(img, dtype=np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_index[0]['index'])
        img1 = np.array(img[0], dtype=np.uint8)
        result_show(img1,label[0],target_decoder(),pred=output[0],input_size=input_size)

def convert_to_tf_lite(model_path,valid_input_size:tuple):
    model = load_model(model_path,compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize=True
    tflite_model = converter.convert()

    start_time = time.strftime('%m_%d_%H_%M')
    name = model_path.split('/')[-1][:-3]

    save_root = "./tflite_model/%s_" % start_time +'_' + name
    os.makedirs(save_root)

    save_path = save_root+"/"+str(valid_input_size)+name+'.tflite'

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    return save_path
input_size = (bg_r,bg_w)


# save_path = convert_to_tf_lite(model_path='models_save/2023_07_02_11_02_50/model_32_0.0589.h5',
#                    valid_input_size=input_size)
tflite_pre(modelpath='./tflite_model/07_02_15_18__model_32_0.0589/(40, 48)model_32_0.0589.tflite',
           dataset_root='./dataset/test',
            input_size=input_size
           )