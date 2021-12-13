import tensorflow
import numpy as np
from metric_learning import CenterLossLayer
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from custom_layers import CutOut


def get_model(mode, image_size, augs, cl = False, num_feats = [16, 32, 64],num_outputs = 2, alpha = 0.5, square_size = 0.3, num_squares = 3, rotation_angle = 0.014):
    
    """
    Simple Model
    """
    
    
    map_fn = {
            'flip': layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            'cutout': CutOut(square_size=square_size, num_squares = num_squares, prob = 0.5),
            'rotation': layers.experimental.preprocessing.RandomRotation(rotation_angle, fill_mode= 'nearest')
    }
    
    inputs = tensorflow.keras.Input(shape=image_size)
    
    if len(augs) > 0:
            
            aug_layers = []
            for l in augs:
                aug_layers.append(map_fn[l])
        
            augment = tensorflow.keras.Sequential(aug_layers)
            x = augment(inputs)
    else:
        x = inputs
            
    
    for num in num_feats:
        x = tensorflow.keras.layers.Conv2D(num, (3,3), activation='relu')(x)
        x = tensorflow.keras.layers.MaxPooling2D(2,2)(x)
    
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    
    feats = tensorflow.keras.layers.Dense(128,  name='feats', activation='relu')(x)
    
    if (mode == 'ml') | (cl):
        feats = tensorflow.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(feats)

    output = tensorflow.keras.layers.Dense(num_outputs, activation = 'softmax',  name = 'cls')(feats)
    
    outs = []
    ins = []
    
    outs.append(output)
    ins.append(inputs)
    
    if mode == 'ml':
        outs.append(feats)
        
    if cl:
        cls_ids = tensorflow.keras.Input(shape=(num_outputs,))
        
        cl_func = CenterLossLayer(alpha = alpha, embedding_shape = feats.shape[1], num_cls= num_outputs)
        
        center_loss = cl_func([feats, cls_ids])
        
        outs.append(center_loss)
        ins.append(cls_ids)
    
    
    
    return tensorflow.keras.Model(inputs=ins, outputs=outs, name = 'mnist_model')