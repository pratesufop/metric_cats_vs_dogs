from utils import load_data, plot_training, umap_plot, gradCAM
from models import get_model
from scheduler import warmup_cosine_method
from metric_learning import triplet_loss, quadruplet_loss, msml, triplet_focal
import tensorflow
import numpy as np
import os
import tensorflow.keras.backend as K
import json
import argparse
# evitar erro de alocação de memória
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

metric_map = {
             'cls': [tensorflow.keras.losses.SparseCategoricalCrossentropy()],
             'triplet': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), triplet_loss()],
             'msml': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), msml()],
             'quadruplet': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), quadruplet_loss()],
             'triplet_focal': [ tensorflow.keras.losses.SparseCategoricalCrossentropy(), triplet_focal(margin= 0.3, sigma = 0.3, lambda_focal = 0.1)]
             }

my_parser = argparse.ArgumentParser()

my_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       help='Path to the experiment config.json')

args = my_parser.parse_args()

config_path = args.Path

with open(config_path) as data_file:
    params = json.load(data_file)

params['outdir'] = os.path.dirname(config_path)

mode = params['mode']
ml_method = params['ml_method']
center_loss = params['center_loss']
bs = params['batch_size']
im_size = params['im_size']
num_epochs = params['num_epochs']
augmentations = params['augmentations']
cls_weight, ml_weight, cl_weight = params['cls_weight'], params['ml_weight'], params['cl_weight']
outdir = params['outdir']

if not os.path.exists(outdir):
    os.makedirs(outdir)
    
train_data, validation_data, num_train_examples = load_data(batch_size = bs, image_res = im_size, md = mode, cl = center_loss, outdir = outdir)
model = get_model(mode= mode, image_size= (im_size,im_size,3), augs = augmentations , cl = center_loss)
model.summary()

lw = [cls_weight]

if mode == 'ml':
    lw.append(ml_weight) 

losses = metric_map[ml_method]
if center_loss:
    lw.append(cl_weight)
    losses.append(lambda y_true,y_pred:  0.5 * K.sum(y_pred, axis=0)) # center loss
    
model.compile(loss= losses,
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights = lw,
             metrics={'cls': tensorflow.keras.metrics.SparseCategoricalAccuracy()})


if (mode == 'ml') | center_loss:
    mon = 'val_cls_sparse_categorical_accuracy'
else:
    mon = 'val_sparse_categorical_accuracy'

checkpoint_filepath = os.path.join(outdir, 'dogs_cats')
model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint( 
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor= mon,
                mode='max',
                save_best_only=True)


lr_method = warmup_cosine_method(total_epochs = num_epochs, batch_size = bs, num_files = num_train_examples, warmup_epoch = 10, base_lr = 0.001)
callback_list = [model_checkpoint_callback, lr_method]

training_output = model.fit(train_data, epochs =  num_epochs, validation_data = validation_data, callbacks= callback_list, shuffle= True)

model.load_weights(checkpoint_filepath)

plot_training(training_output, mode= mode, cl = center_loss, outdir = outdir)

outputs = model.evaluate(validation_data)
outputs = dict([(m, out) for out, m in zip(outputs, model.metrics_names)])

if (mode == 'ml') | center_loss:
    acc = outputs['cls_sparse_categorical_accuracy']
else:
    acc = outputs['sparse_categorical_accuracy']
    
print('Test Acc. %.2f' % (100*acc))

umap_plot(model, validation_data, outdir = outdir)

# sample an image to plot using the GradCAM heatmap
x, y = next(iter(validation_data))
img = x['input_1'][np.random.choice(np.arange(len(x)))]

gradCAM(img, model, outdir = outdir)

output_params = {}
output_params['acc'] =  '%.2f' % (100*acc)

with open(os.path.join(outdir, 'result.json'), 'w') as data_file:
    json.dump(output_params, data_file)
