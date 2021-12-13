import tensorflow
import tensorflow.keras.backend as K

def squared_dist(A, B):

    row_norms_A = tensorflow.reduce_sum(tensorflow.square(A), axis=1)
    row_norms_A = tensorflow.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tensorflow.reduce_sum(tensorflow.square(B), axis=1)
    row_norms_B = tensorflow.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tensorflow.matmul(A, tensorflow.transpose(B)) + row_norms_B

def get_live_live_mask(labels):
    
    tempA = tensorflow.tile(tensorflow.expand_dims(tensorflow.transpose(labels), axis=1), tensorflow.stack([tensorflow.constant(1), tensorflow.shape(labels)[0]], axis=0)  )
    tempB = tensorflow.tile(tensorflow.expand_dims(labels, axis=0), tensorflow.stack([tensorflow.shape(labels)[0], tensorflow.constant(1)], axis=0)  )
    return tempA*tempB # mask live-live
    
def get_spoof_spoof_mask(labels):
    
    tempA = tensorflow.tile(tensorflow.expand_dims(tensorflow.transpose(1 - labels), axis=1), tensorflow.stack([tensorflow.constant(1), tensorflow.shape(labels)[0]], axis=0)  )
    tempB = tensorflow.tile(tensorflow.expand_dims((1 - labels), axis=0), tensorflow.stack([tensorflow.shape(labels)[0], tensorflow.constant(1)], axis=0)  )
    return tempA*tempB # mask live-live



class CenterLossLayer(tensorflow.keras.layers.Layer):

    def __init__(self,alpha, embedding_shape, num_cls, **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.embedding = embedding_shape
        self.num_cls = num_cls

    def build(self,input_shape):
        self.centers=self.add_weight(name="centers",
                                    shape=(self.num_cls,self.embedding),
                                    initializer="uniform",
                                    trainable=False)
        super().build(input_shape)

    def call(self,x,mask=None):
        #x[0]:N*2 x[1]:N*10 centers:10*2
        delta_centers=K.dot(K.transpose(x[1]),K.dot(x[1],self.centers)-x[0])
        centers_count=K.sum(K.transpose(x[1]),axis=-1,keepdims=True)+1
        delta_centers/=centers_count
        
        new_centers=self.centers-self.alpha*delta_centers
        self.add_update((self.centers,new_centers),x)

        self.result=x[0]-K.dot(x[1],self.centers)
        
        self.result=K.sum(self.result**2,axis=1,keepdims=True)
        
        return self.result

    def compute_output_shape(self,input_shape):
        return K.int_shape(self.result)

def quadruplet_loss(margin= 0.3):
    
    '''
    
    Metric Learning using quadruplet. 

    '''
    def calculate_quadloss(y_true, y_pred):
        
        margin_tensor = tensorflow.constant([margin])

        labels = tensorflow.cast(y_true, dtype=tensorflow.float32)[:,0]

        #l = tensorflow.where(tensorflow.math.equal(labels, 1))
        dist = squared_dist(y_pred, y_pred)
        dist = tensorflow.cast(dist,dtype=tensorflow.float32)
        
        # getting the masks
        maskll = get_live_live_mask(labels)
        maskss = get_spoof_spoof_mask(labels)
        masksl = tensorflow.ones_like(maskss) - maskss - maskll # mask between spoof and lives
    
        maskss = tensorflow.cast(maskss,dtype=tensorflow.float32)*(1 - tensorflow.eye(tensorflow.shape(maskss)[0], dtype=tensorflow.float32))
        
        # distance between two live samples
        dist_max = tensorflow.math.reduce_max(
                    tensorflow.math.multiply(dist, tensorflow.cast(tensorflow.math.equal(maskll,1), dtype=tensorflow.float32)),
                    axis=1)

        # distance between live and spoof samples
        dist_min = tensorflow.math.reduce_min(
                    tensorflow.where(tensorflow.math.equal(masksl,1), dist, tensorflow.cast(tensorflow.math.equal(masksl,0), dtype=tensorflow.float32)*dist_max))
        
            
        # distance between spoof and spoof
        dist_negpair = tensorflow.reduce_min(tensorflow.gather_nd(dist, tensorflow.where(maskss == 1)), axis= None)

        loss1 = tensorflow.reduce_sum(tensorflow.multiply(
                                tensorflow.math.maximum(
                                    dist_max - dist_min + margin_tensor, tensorflow.constant([0], dtype=tensorflow.float32)
                                ), 
                                tensorflow.cast(labels,dtype=tensorflow.float32)))/tensorflow.reduce_sum(tensorflow.cast(labels,dtype=tensorflow.float32))

        loss2 = tensorflow.reduce_sum(tensorflow.multiply(
                                tensorflow.math.maximum(
                                    dist_max - dist_negpair + margin_tensor, tensorflow.constant([0], dtype=tensorflow.float32)
                                ), 
                                tensorflow.cast(labels,dtype=tensorflow.float32)))/tensorflow.reduce_sum(tensorflow.cast(labels,dtype=tensorflow.float32))

        loss = loss1 + loss2  
        
        return loss
    return calculate_quadloss

def triplet_focal(margin= 0.3, sigma = 0.5, lambda_focal = 1):
    
    '''
    
    Metric Learning usando triplet focal (paper: Deep Anomaly Detection for Generalized Face Anti-Spoofing)
    
    '''

    def calculate_triloss(y_true, y_pred):
        
        margin_tensor = tensorflow.constant([margin])
        
        labels = tensorflow.cast(y_true, dtype=tensorflow.float32)[:,0]

        dist = squared_dist(y_pred, y_pred)
        dist = tensorflow.cast(dist,dtype=tensorflow.float32)

        # getting the masks
        maskll = get_live_live_mask(labels)
        maskss = get_spoof_spoof_mask(labels)
        masksl = tensorflow.ones_like(maskss) - maskss - maskll # mask between spoof and lives

        # distance between two live samples
        dist_max = tensorflow.math.reduce_max(
                    tensorflow.math.multiply(dist, tensorflow.cast(tensorflow.math.equal(maskll,1), dtype=tensorflow.float32)), 
                    axis=1)

        # distance between live and spoof samples
        dist_min = tensorflow.math.reduce_min(
                    tensorflow.where(tensorflow.math.equal(masksl,1), dist, tensorflow.cast(tensorflow.math.equal(masksl,0), dtype=tensorflow.float32)*dist_max), 
                    axis=1)
        
        
        loss_metric_soft = -tensorflow.reduce_sum(tensorflow.math.log(tensorflow.math.exp(dist_max)/(tensorflow.math.exp(dist_max) + tensorflow.math.exp(dist_min))))/tensorflow.reduce_sum(tensorflow.cast(labels,dtype=tensorflow.float32))
        
        
        loss_triplet_focal = tensorflow.reduce_sum(tensorflow.multiply(
                                                        tensorflow.math.maximum(
                                                            tensorflow.math.exp(dist_max/sigma) - tensorflow.math.exp(dist_min/sigma)  + margin_tensor, tensorflow.constant([0], dtype=tensorflow.float32)
                                                        ), 
                                                        tensorflow.cast(labels,dtype=tensorflow.float32)))/tensorflow.reduce_sum(tensorflow.cast(labels,dtype=tensorflow.float32))
                                    
        return lambda_focal*loss_triplet_focal + loss_metric_soft
    
    return calculate_triloss





def triplet_loss(margin= 0.3):
    
    '''
    
    Metric Learning using triplet

    '''

    def calculate_triloss(y_true, y_pred):
        
        margin_tensor = tensorflow.constant([margin])
        
        labels = tensorflow.cast(y_true, dtype=tensorflow.float32)[:,0]
        
        dist = squared_dist(y_pred, y_pred)
        dist = tensorflow.cast(dist,dtype=tensorflow.float32)

        # getting the masks
        maskll = get_live_live_mask(labels)
        maskss = get_spoof_spoof_mask(labels)
        masksl = tensorflow.ones_like(maskss) - maskss - maskll # mask between spoof and lives

        # distance between two live samples
        dist_max = tensorflow.math.reduce_max(
                    tensorflow.math.multiply(dist, tensorflow.cast(tensorflow.math.equal(maskll,1), dtype=tensorflow.float32)), 
                    axis=1)

        # distance between live and spoof samples
        dist_min = tensorflow.math.reduce_min(
                    tensorflow.where(tensorflow.math.equal(masksl,1), dist, tensorflow.cast(tensorflow.math.equal(masksl,0), dtype=tensorflow.float32)*dist_max), 
                    axis=1)

        loss = tensorflow.reduce_sum(tensorflow.multiply(
                                                        tensorflow.math.maximum(
                                                            dist_max - dist_min + margin_tensor, tensorflow.constant([0], dtype=tensorflow.float32)
                                                        ), 
                                                        tensorflow.cast(labels,dtype=tensorflow.float32)))/tensorflow.reduce_sum(tensorflow.cast(labels,dtype=tensorflow.float32))
                                
        return loss
    
    return calculate_triloss


def msml(margin= 0.3):
    
    '''
    Margin-Sample Mining Loss
    '''

    def calculate_msml(y_true, y_pred):
        
        margin_tensor = tensorflow.constant([margin])
        
        labels = tensorflow.cast(y_true, dtype=tensorflow.float32)[:,0]
        
        dist = squared_dist(y_pred, y_pred)
        dist = tensorflow.cast(dist,dtype=tensorflow.float32)
         
        # getting the masks
        maskll = get_live_live_mask(labels)
        maskss = get_spoof_spoof_mask(labels)
        masksl = tensorflow.ones_like(maskss) - maskss - maskll # mask between spoof and lives

        # distance between two live samples
        dist_max = tensorflow.math.reduce_max(
                    tensorflow.math.multiply(dist, tensorflow.cast(tensorflow.math.equal(maskll,1), dtype=tensorflow.float32)))

        # distance between two live samples
        dist_min = tensorflow.math.reduce_min(
                    tensorflow.where(tensorflow.math.equal(masksl,1), dist, tensorflow.cast(tensorflow.math.equal(masksl,0), dtype=tensorflow.float32)*dist_max))

        loss = tensorflow.math.maximum(dist_max - dist_min + margin_tensor, tensorflow.constant([0], dtype=tensorflow.float32)   )
                                                                                         
        return loss
    
    return calculate_msml