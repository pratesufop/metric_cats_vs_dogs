from utils import load_minst_data, plot_training, umap_plot
from models import get_model
from scheduler import warmup_cosine_method
from metric_learning import triplet_loss, quadruplet_loss, msml, triplet_focal
import tensorflow
import numpy as np

metric_map = {
             'cls': [tensorflow.keras.losses.SparseCategoricalCrossentropy()],
             'triplet': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), triplet_loss()],
             'msml': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), msml()],
             'quadruplet': [tensorflow.keras.losses.SparseCategoricalCrossentropy(), quadruplet_loss()],
             'triplet_focal': [ tensorflow.keras.losses.SparseCategoricalCrossentropy(), triplet_focal(margin= 0.3, sigma = 0.3, lambda_focal = 0.1)]
             }


mode = 'ml'
ml_method = 'msml'

x_train, y_train, x_test, y_test = load_minst_data()
model = get_model(mode= mode)
model.summary()

model.compile(loss= tensorflow.keras.losses.SparseCategoricalCrossentropy(),
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights=[1],
            metrics=['accuracy'])


checkpoint_filepath = './mnist'
model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(loss = metric_map[ml_method],
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='accuracy',
                mode='max',
                loss_weights = [1, 0.001],
                save_best_only=True)


lr_method = warmup_cosine_method(total_epochs = 30, batch_size = 128, num_files = len(x_train), warmup_epoch = 10, base_lr = 0.001)
callback_list = [model_checkpoint_callback, lr_method]

if mode == 'ml':
    outputs = [y_train, np.random.rand(y_train.shape[0],1)]
else:
    outputs = y_train

training_output = model.fit(x_train, y_train, validation_split=0.2, batch_size = 128 , 
                                            epochs =  30, steps_per_epoch =  int(len(x_train) / 128 ), 
                                            callbacks= callback_list, shuffle= True)

loss, acc = model.evaluate(x_test,y_test)
print('Test Acc. %.2f' % (100*acc))

umap_plot(model, x_test, y_test)
