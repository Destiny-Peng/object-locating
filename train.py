from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
from training.callbacks import *
from training.loss_function import SSD_loss
from training.metrics import Recall
import time
from nets.ObjectLocatingModel import ShuoShuoNet
from parameters import *

if __name__=='__main__':
    batch_size =128
    img_num=5
    BN_momentum=0.99
    input_shape = (bg_r, bg_w, 3)

    train_dataset = Fast_dataset(
        imgs_path='./dataset/train/images',
        labels_path='./dataset/train/labels.npy',
        batch_size=batch_size,
        input_size=(bg_r,bg_w)
    )
    valid_dataset =  Fast_dataset(
        imgs_path='./dataset/valid/images',
        labels_path='./dataset/valid/labels.npy',
        batch_size=batch_size,
        input_size=(bg_r,bg_w)
    )


    model = ShuoShuoNet(input_shape=input_shape,
                        alpha=0.35,
                        FeatureMap_shape=(grid_r,grid_w)
                        ).model()
    for layer in model.layers:
        if type(layer) == type(keras.layers.BatchNormalization()):
            layer.momentum = BN_momentum
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=SSD_loss(),
                  metrics=[Recall(grid=(grid_r,grid_w))])

    save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))

    save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_loss:.4f}.h5",
                                                   save_best_only=True, monitor='val_loss')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    # tensorboard_callback = MyTensorboardCallback('logs',input_size=(bg_r,bg_w))

    callback_list=[save_weights,reduce_lr,early_stop]
    hist = model.fit(train_dataset,
                     epochs=100,
                     workers=8,
                     validation_data=valid_dataset,
                     callbacks=callback_list
                     )



