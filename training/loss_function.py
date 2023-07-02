import tensorflow as tf
from tensorflow import keras
from parameters import *
"""
损失分为两部分，位置损失和置信度损失
"""
class Conf_loss(tf.losses.Loss):
    def __init__(self,):
        super().__init__(
            reduction="none", name="Conf_loss"
        )
    def call(self, y_true, y_pred):
        """
        y shape : (batch size ,12)
        use BinaryCrossentropy
        return (bs)
        """
        # keras 会自动求平均，所以最后要乘回一个l
        _,l = y_true.shape

        conf_loss = keras.losses.BinaryCrossentropy(reduction="none")
        return tf.multiply(conf_loss(y_true, y_pred),l)
class Angle_Loss(tf.losses.Loss):
  def __init__(self):
    super().__init__(reduction=tf.losses.Reduction.NONE, name='angle_loss')

  def call(self, y_true, y_pred):
    angle_loss = tf.square(y_true - y_pred) # 取预测与真实角度的绝对值差作为误差
    return tf.reduce_mean(angle_loss, axis=-1) # 沿着最后一个轴(batch内的样本)取平均
class Smooth_L1_loss(tf.losses.Loss):
    def __init__(self, delta=1):
        super().__init__(
            reduction="none", name="Smooth_L1_loss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        """
        input shape : (batch size ,12,2)
        return shape : (batch size,12)
        """
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)#对x,y两个坐标偏移量求和

class Loc_loss(tf.losses.Loss):
    def __init__(self,grid=(grid_r,grid_w)):
        super().__init__(
            reduction="none", name="Loc_loss"
        )
        self.smoothl1 = Smooth_L1_loss()
        self.grid = grid
        self.grid_r,self.grid_w = grid
    def _mask(self,y_true_conf,y_pred_conf):
        """
        get the mask of Loc_loss
        the mask use in mask*smoothL1
        y shape : (batch size ,12)
        mask的值当且仅当y_true 与 y_pred 都有目标时为1
        return a tensor (batch size,12)
        """
        y_pred_conf = tf.where(y_pred_conf>0.5,1.0,0.)
        mask = tf.multiply(y_true_conf,y_pred_conf)
        # mask = tf.repeat(mask,repeats=2,axis=-1)
        return mask
    def call(self, y_true, y_pred):
        """
        input shape : (batch size ,24)
        return shape : (batch size)
        """


        y_true_conf = y_true[:,2*self.grid_w*self.grid_r :3*self.grid_w*self.grid_r]
        y_pred_conf = y_pred[:,2*self.grid_w*self.grid_r :3*self.grid_w*self.grid_r]

        #mask shape:(bs,12)
        mask = self._mask(y_true_conf,y_pred_conf)

        y_true_loc = y_true[:,:2*self.grid_w*self.grid_r]
        y_pred_loc = y_pred[:,:2*self.grid_w*self.grid_r]

        y_true_loc = tf.reshape(y_true_loc,shape=(-1,self.grid_w*self.grid_r,2))
        y_pred_loc = tf.reshape(y_pred_loc,shape=(-1,self.grid_w*self.grid_r,2))

        #loss shape:(bs,12)
        loss = self.smoothl1(y_true_loc,y_pred_loc)

        loss = tf.multiply(loss,mask)

        #对所以格子的位置损失求和
        loss = tf.reduce_sum(loss,axis=-1)
        return loss
class SSD_loss(tf.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, grid=(grid_r,grid_w)):
        super().__init__(reduction="auto", name="SSD_loss")
        self._loc_loss = Loc_loss()
        self._conf_loss = Conf_loss()
        self._angle_loss = Angle_Loss()
        self.alpha= alpha
        self.beta = beta
        self.grid = grid
        self.grid_w,self.grid_r = grid
    def call(self, y_true, y_pred):
        """
        N : number of tagert in y_true
        input shape : (batch size,36)
        """

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_true = tf.reshape(y_true,shape=(-1,4*self.grid_w*self.grid_r))

        y_true_conf = y_true[:,2*self.grid_w*self.grid_r :3*self.grid_w*self.grid_r]
        y_pred_conf = y_pred[:,2*self.grid_w*self.grid_r :3*self.grid_w*self.grid_r]
        N = tf.reduce_sum(y_true_conf, axis=-1)

        y_true_angle = y_true[:,3*self.grid_w*self.grid_r :]
        y_pred_angle = y_pred[:,3*self.grid_w*self.grid_r :]
        #return (batch size)
        loc_loss = self._loc_loss(y_true,y_pred)
        conf_loss = self._conf_loss(y_true_conf,y_pred_conf)
        angle_loss = self._angle_loss(y_true_angle,y_pred_angle)
        loss = 1/N*(loc_loss+self.alpha*conf_loss+self.beta*angle_loss)
        return loss


if __name__ == "__main__":
    Loss = SSD_loss()
    y_true = tf.random.normal((4, 4 * grid_r * grid_w), dtype=tf.float32)
    y_pred = tf.random.normal((4, 4 * grid_r * grid_w), dtype=tf.float32)
    loss=Loss(y_true,y_pred)
    print(loss)
