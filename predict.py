from tensorflow import keras
from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
from utils.util import *
from utils.target_utils import target_decoder
def predict(model,dataset):
    for (img, label) in dataset:
        img1 = np.array(img[0], dtype=np.uint8)
        # print(label)
        result_show(img1,label[0],target_decoder(),model)



if __name__ =='__main__':

    model = keras.models.load_model('models_save/2023_07_01_21_06_38/model_36_0.0248.h5',compile=False)
    # model.summary()
    batch_size =1
    img_num=5
    # valid_dataset = RandomTarget_dataset(root = r'D:\Python Project\Project1\venv\dataset',
    #                       batch_size=batch_size,
    #                       bg_r=96,
    #                       bg_w=128,
    #                       bg_root=r'D:\Python Project\object-locating\background',
    #                       img_num=img_num,
    #                       Chinese_path=True,
    #                       valid=True
    #                      )
    valid_dataset =  Fast_dataset(
        imgs_path='./dataset/valid/images',
        labels_path='./dataset/valid/labels.npy',
        batch_size=batch_size,
        input_size=(bg_r,bg_w)
    )
    predict(model,valid_dataset)