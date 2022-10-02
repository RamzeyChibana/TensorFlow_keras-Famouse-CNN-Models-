import tensorflow as tf
from tensorflow import keras
from keras import layers,regularizers

class VggBlock(layers.Layer):
    def __init__(self,num_cn,channels):
        super(VggBlock,self).__init__()
        self.num_cn=num_cn
        self.layers=[]
        for i in range(num_cn):
            self.layers.append(tf.keras.layers.Conv2D(channels,3,kernel_regularizer=regularizers.l2(),activation=tf.nn.relu))
        self.pool=tf.keras.layers.MaxPool2D(strides=2)
    
    def call(self,inputs):
        x=inputs
        for i in range(self.num_cn):
            x=self.layers[i](x)
        x=self.pool(x)
        return x


class VGG_19(keras.Model):
    def __init__(self) :
        super(VGG_19,self).__init__()
        self.vgg1=VggBlock(2,64)
        self.vgg2=VggBlock(2,128)
        self.vgg3=VggBlock(4,256)
        self.vgg4=VggBlock(4,512)
        self.vgg5=VggBlock(4,512)
        self.fc1=layers.Dense(4096,activation=tf.nn.relu)
        self.fc2=layers.Dense(4096,activation=tf.nn.relu)
        self.outputlayer=layers.Dense(1,activation=tf.nn.sigmoid)



    def call(self,inputs):
        x=self.vgg1(inputs)
        x=self.vgg2(x)
        x=self.vgg3(x)
        x=self.vgg4(x)
        x=layers.Flatten()(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.outputlayer(x)
        return x
    def model(self):
        input=keras.Input((244,244,3))
        return keras.Model(inputs=[input],outputs=self.call(input))