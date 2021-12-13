from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K



def cutout(images, square_size, num_squares, prob):
    
    images = tf.cast(images, tf.float32)
    
    img_shape = images.shape
    _, h, w, c = img_shape[0], img_shape[1], img_shape[2], img_shape[3]

    bs = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)
    
    prob_img = tf.cast(tf.random.uniform([bs],0,1) <= prob, tf.int32)
    prob_img = tf.cast(tf.tile(tf.reshape(prob_img, [-1,1,1]), [1,h,w]), tf.int32)
    
    ssp = tf.cast(tf.math.ceil(h * square_size), tf.int32)

    coords_x = tf.random.uniform([bs, num_squares], 
                                 minval=0, 
                                 maxval= (h - ssp), 
                                 dtype=tf.int32) 

    coords_x = tf.linspace(coords_x, coords_x + ssp - 1, ssp)

    coords_x = tf.cast(tf.transpose(coords_x), tf.int32)

    coords_y = tf.random.uniform([bs, num_squares], 
                                 minval=0, 
                                 maxval= (h - ssp), 
                                 dtype=tf.int32) 

    
    coords_y = tf.linspace(coords_y, coords_y + ssp - 1, ssp)

    coords_y = tf.cast(tf.transpose(coords_y), tf.int32)

    grid_y = tf.reshape(tf.tile(coords_y, [1,1,ssp]), 
                        (bs, num_squares, 1, ssp * ssp))

    grid_y = tf.transpose(tf.reshape(grid_y, 
                                     (bs, num_squares, ssp, ssp)), 
                          (0, 1, 3, 2))

    grid_x = tf.reshape(tf.tile(coords_x, [1,1,ssp]), 
                        (bs, num_squares, 1, ssp * ssp))

    grid_x = tf.reshape(grid_x, (bs, num_squares, ssp, ssp))

    grid = tf.stack([grid_y, grid_x], axis=0)

    grid = tf.reshape(tf.transpose(grid, (1, 4, 3, 2, 0)), 
                      (bs, ssp * ssp * num_squares, 2))

    batch_indices = tf.reshape(tf.tile(tf.range(0, bs), 
                                       [ssp * ssp * num_squares]), 
                               (ssp * ssp * num_squares, bs))

    batch_indices = tf.reshape(tf.transpose(batch_indices), 
                               (bs, ssp * ssp * num_squares, 1))

    grid = tf.concat([batch_indices, grid], axis=2)

    masks = tf.scatter_nd(grid[tf.newaxis,...], 
                            tf.ones([1,bs,ssp*ssp*num_squares]) * -1, 
                            shape=(bs, h, w)) +1

    masks = tf.clip_by_value(masks, 0, 1)
    
    masks = tf.where(tf.cast(prob_img, tf.bool), masks, tf.ones_like(masks))

    random_image = tf.random.uniform([bs, h,w,c], 
                                 minval=0, 
                                 maxval= 1, 
                                 dtype=tf.float32)

    return images * masks[..., tf.newaxis] + random_image*(1 - masks[..., tf.newaxis])


class CutOut(layers.Layer):
    def __init__(self, square_size=0.3, num_squares = 2, prob = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.square_size, self.num_squares, self.prob = square_size, num_squares, prob
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'square_size': self.square_size,
            'num_squares': self.num_squares,
            'prob': self.prob
        })
        return config
    
    def call(self, x, training=None ):
        if not training:
            return x
        else:
            return cutout(x, self.square_size, self.num_squares, self.prob)