import tensorflow
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import umap as umap
import umap.plot
import tensorflow_datasets as tfds
import os

IMAGE_RES = 224
mode = 'ml'
center_loss = False

def format_image(image, label):
    image = tensorflow.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    
    if center_loss:
        ins = {'input_1': image, 'input_2': tensorflow.one_hot(label, depth = 2)}
    else:
        ins = {'input_1': image}
        
    if (mode == 'ml') & center_loss:
        out = {'cls': label, 'lambda': label, 'center_loss_layer': label}
    elif mode == 'ml':
        out = {'cls': label, 'lambda': label}
    elif center_loss:
        out = {'cls': label, 'center_loss_layer': label}
    else:
        out = {'cls': label}
        
    return ins, out

def load_data(batch_size, image_res, outdir, md = 'ml', cl = False):
    
    global IMAGE_RES, mode, center_loss
    
    center_loss = cl
    mode = md
    IMAGE_RES = image_res
    
    (train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split = ('train[:80%]', 'train[80%:]'),
    with_info = True,
    as_supervised = True
    )
    num_examples = info.splits['train'].num_examples

    train_batches = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(batch_size).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(batch_size).prefetch(1)
    
    class_names = np.array(info.features['label'].names)
    image_batch, label_batch = next(iter(train_batches))
    
    
    image_batch = image_batch['input_1']
    lb = label_batch['cls']
        
    label_class_names = class_names[lb]
    
    plt.figure(figsize=(10,9))
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(image_batch[n])
        color = "blue" if lb[n] == 1 else "red"
        plt.title(label_class_names[n].title(), color=color)
        plt.axis('off')
        _= plt.suptitle("Model predictions (blue: dog, red: cat)")
        
    plt.savefig(os.path.join(outdir,'exemplos.png'))
    
    return train_batches, validation_batches, num_examples


def plot_training(training_output, outdir , mode = 'cls', cl = False):
    
    plt.figure(figsize=(20,10))

    plt.subplot(121)
    if (mode == 'ml') | (cl):
        acc_train = 'cls_sparse_categorical_accuracy'
        acc_val = 'val_cls_sparse_categorical_accuracy'

    else:
        acc_train = 'sparse_categorical_accuracy'
        acc_val = 'val_sparse_categorical_accuracy'

    
    plt.plot(training_output.history[acc_train])
    plt.plot(training_output.history[acc_val])
    plt.title('Acc. Epochs')
    plt.legend(['Acc','Val. Acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Acc')

    plt.subplot(122)
    plt.plot(training_output.history['loss'])
    plt.plot(training_output.history['val_loss'])
    plt.title('Loss Epochs')
    plt.legend(['Loss','Val. Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.savefig(os.path.join(outdir,'training.png'))
    

def gradCAM(img , model, outdir ):
    
    last_conv_layer = model.get_layer('conv2d_2')
    
    if len(model.inputs) > 1:
        inp = model.inputs[0]
    else:
        inp = model.inputs
        
    last_conv_layer_model = tensorflow.keras.Model(inp, last_conv_layer.output)

    classifier_input = tensorflow.keras.Input(shape=last_conv_layer.output.shape[1:])

    classifier_layer_index = ['global_average_pooling2d','feats','cls']
    x = classifier_input
    for layer_index in classifier_layer_index:
        x = model.get_layer(layer_index)(x)
    classifier_model = tensorflow.keras.Model(classifier_input, x)

    with tensorflow.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(tensorflow.expand_dims(img, axis=0))
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tensorflow.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tensorflow.reduce_mean(grads, axis=(0,1,2))
    pooled_grads = pooled_grads.numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    # Relu
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img.shape[0],img.shape[1]))

    heatmap = np.uint8(255 * np.abs(heatmap))

    plt.figure(figsize=(20,20))

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(heatmap)
    
    plt.savefig(os.path.join(outdir,'gradCAM.png'))
    
def umap_plot(model, validation_data, outdir):
    
    feat_model = tensorflow.keras.Model(inputs = model.inputs, outputs= model.get_layer('feats').output)
    all_feats = feat_model.predict(validation_data)
    
    y_val = list(map(lambda x: x[1]['cls'], validation_data))
    y_val = np.array(tensorflow.concat(y_val, axis=0))

    classes = np.unique(y_val)
    embedding = umap.UMAP(n_neighbors=11).fit_transform(np.array(all_feats))

    _, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*embedding.T, s=20.0, c=y_val, cmap='jet_r', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP Embedding', fontsize=14)
    cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5)
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outdir,'umap.png'))
