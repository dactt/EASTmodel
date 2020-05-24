import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Conv2D,concatenate,Lambda,Input,multiply,add,ZeroPadding2D,Activation,Layer,MaxPool2D,Dropout,BatchNormalization
from keras import regularizers

RESIZE_FACTOR = 2

def uppool(x):
    return tf.compat.v1.image.resize(x,size=[tf.shape(x)[1]*RESIZE_FACTOR,tf.shape(x)[2]*RESIZE_FACTOR])
class EAST_model:
    def __init__(self,input_size = 512):
        input_image = Input(shape=[None,None,3],name="input_image")
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')
        resnet = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)

        # backborn resnet
        x = resnet.get_layer(index=174).output

        # state 1 feature-merging
        x = Lambda(uppool, name='uppool_1')(x)
        x = concatenate([x, resnet.get_layer(index=142).output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        # state 2 feature-merging
        x = Lambda(uppool, name='uppool_2')(x)
        x = concatenate([x, resnet.get_layer(index= 80).output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        # state 3 feature-merging
        x = Lambda(uppool, name='uppool_3')(x)
        x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(resnet.get_layer(index = 38).output)], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        # state 4 feature-merging
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        # output
        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x) 
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')
        
        model = Model(inputs=[input_image, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])


        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map