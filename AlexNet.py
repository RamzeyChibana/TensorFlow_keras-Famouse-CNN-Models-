import tensorflow as tf
from tensorflow import keras
from keras import layers,regularizers





class CNNBlock(layers.Layer):
    def __init__(self,filter_size,channels,padding="same",filter_stride=1,pool_info=None):
        super(CNNBlock,self).__init__()
        self.dopool=False
        self.conv=layers.Conv2D(channels,filter_size,filter_stride,padding=padding)
        self.batch=layers.BatchNormalization()
        if pool_info != None:
            self.pooling=layers.MaxPool2D(pool_info[0],pool_info[1])
            self.dopool=True
    
    def call(self,inputs,training=False):
        x=self.conv(inputs)
        x=self.batch(x,training=training)
        x=tf.nn.relu(x)
        if self.dopool:
            x=self.pooling(x)
        return x


class FullyConnected(layers.Layer):
    def __init__(self,units,lambda_reg=0.01,drob_rate=0.3):
        super(FullyConnected,self).__init__()
        self.layer=layers.Dense(units,kernel_regularizer=regularizers.l2(lambda_reg))
        self.batch=layers.BatchNormalization()
        self.drob=layers.Dropout(drob_rate)
    def call(self, inputs, training=False):
        x=self.layer(inputs)
        x=self.batch(x,training=training)
        x=tf.nn.relu(x)
        x=self.drob(x,training=training)
        return x



class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.cnn1=CNNBlock((11,11),96,'valid',4,((3,3),2))
        self.cnn2=CNNBlock((5,5),256,pool_info=((3,3),2))
        self.cnn3=CNNBlock((3,3),384)
        self.cnn4=CNNBlock((3,3),384)
        self.cnn5=CNNBlock((3,3),256,pool_info=((3,3),2))
        self.fc6=FullyConnected(4096,drob_rate=0.4)
        self.fc7=FullyConnected(4096,drob_rate=0.4)
        self.outputlayer=layers.Dense(1,activation="sigmoid")
    def call(self,inputs,training=False):
        x=self.cnn1(inputs,training)
        x=self.cnn2(x,training=training)
        x=self.cnn3(x,training=training)
        x=self.cnn4(x,training=training)
        x=self.cnn5(x,training=training)
        x=layers.Flatten()(x)
        x=self.fc6(x,training=training)
        x=self.fc7(x,training=training)
        x=self.outputlayer(x,training=training)