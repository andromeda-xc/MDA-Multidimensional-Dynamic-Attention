

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# from keras.utils.vis_utils import plot_model
from tqdm import tqdm
import keras.backend as K

import os
import random
import numpy as np
# import tensorflow_probability as tfp

# tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
tf.keras.backend.clear_session()
tf.random.set_seed(111)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# K.set_session(sess)

def reset_seed(seed):
    # tf.disable_v2_behavior()
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    # tf.set_random_seed(seed)
    # config = tf.ConfigProto()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # K.set_session(sess)

def compile(model,params):
    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=float(params['learning_rate']) )#,weight_decay = True)
    model.compile( loss=loss, optimizer=optimizer, metrics=['mae']) #'mse')
    return model

class AmplificationLayer(layers.Layer):
    def __init__(self, exponent, **kwargs):
        super(AmplificationLayer, self).__init__(**kwargs)
        self.exponent = exponent

    def call(self, inputs):
        amplified_output = tf.pow(inputs, self.exponent)
        return amplified_output


class Attention(  layers.Layer):
    # by : https://github.com/philipperemy/keras-attention ; with some modifications 
    SCORE_LUONG = 'luong'
    SCORE_BAHDANAU = 'bahdanau'

    def __init__(self, units: int = 128, horizon = 0, score: str = 'luong',return_score=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        if score not in {self.SCORE_LUONG, self.SCORE_BAHDANAU}:
            raise ValueError(f'Possible values for score are: [{self.SCORE_LUONG}] and [{self.SCORE_BAHDANAU}].')
        self.units = units
        self.score = score
        self.return_score=return_score
        self.horizon = horizon #used for dynamic attetnion embedding

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        if len(input_shape) >1:
          self.hNorm =   tf.keras.layers.LayerNormalization()    
          self. intermidiate_projector =  tf.keras.layers.Dense(self.units ,use_bias=False) 
        input_dim = int(input_shape[0][-1])
        with K.name_scope(self.name ):
            # W in W*h_S.
            if self.score == self.SCORE_LUONG:
                self.luong_w = tf.keras.layers.Dense(input_dim, use_bias=False, name='luong_w')
                self.luong_dot = tf.keras.layers.Dot(axes=[1, 2], name='attention_score')
            else:

                self.bahdanau_v = tf.keras.layers.Dense(1, use_bias=False, name='bahdanau_v')
                self.bahdanau_w1 = tf.keras.layers.Dense(input_dim, use_bias=False, name='bahdanau_w1')
                self.bahdanau_w2 = tf.keras.layers.Dense(input_dim, use_bias=False, name='bahdanau_w2')
                self.bahdanau_repeat = tf.keras.layers.RepeatVector(input_shape[0][1])
                self.bahdanau_tanh = tf.keras.layers.Activation('tanh', name='bahdanau_tanh')
                self.bahdanau_add = tf.keras.layers.Add()

            self.h_t = tf.keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')

            self.softmax_normalizer = tf.keras.layers.Activation('softmax', name='attention_weight')

            self.dot_context = tf.keras.layers.Dot(axes=[1, 1], name='context_vector')

            self.concat_c_h = tf.keras.layers.Concatenate(name='attention_output')
            self.w_c = tf.keras.layers.Dense(self.units, use_bias=False, activation='tanh', name='attention_vector') 

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.units 

    def call(self, inputs, training=None, **kwargs):
        print('\n\n -------------------------- attention starts --------\n\n')
        

        h_s = inputs[0]
        h_t = self.h_t(h_s)
        if len(inputs)>1:

            d_c = inputs[1] #this is for the TAU
            d_c = self.intermidiate_projector ( tf.expand_dims(d_c, axis=-1))
            d_c_broadcasted = tf.expand_dims(d_c, axis=1)
            d_c = tf.tile(d_c_broadcasted, [1, tf.shape(h_s)[1], 1])

            h_s = tf.keras.layers.Add()([d_c,h_s] )#h_t
            
        if self.score == self.SCORE_LUONG:
            # Luong's multiplicative style.
            score = self.luong_dot([h_t, self.luong_w(h_s)])


        else:
            # Bahdanau's additive style.
            self.bahdanau_w1(h_s)
            a1 = self.bahdanau_w1(h_t)
            a2 = self.bahdanau_w2(h_s)
            a1 = self.bahdanau_repeat(a1)
            score = self.bahdanau_tanh(self.bahdanau_add([a1, a2]))
            score = self.bahdanau_v(score)
            score = K.squeeze(score, axis=-1)


        alpha_s = self.softmax_normalizer(score)

        context_vector = self.dot_context([h_s, alpha_s])
        
        a_t = self.w_c(self.concat_c_h([context_vector, h_t]))

        # if len(inputs)>1:
        #     print('TA context > a_t',a_t)

        if self.return_score:
            return a_t , alpha_s
        return a_t

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'units': self.units, 'score': self.score})
        return config

class DynamicEmbeddings(layers.Layer):
    def __init__(self,  units,**kwargs ):
        super(DynamicEmbeddings, self).__init__(**kwargs)
        self.units =  units

    def build(self,input_shape):

        #D_C learner
        self.hX = Attention(units=self.units)
        self.G = tf.keras.layers.LSTM(16, activation='tanh',return_sequences=True,return_state=False)

    def call(self, inputs):


        return self.hX( [self.G(inputs) ])

class VA(tf.keras.layers.Layer):
    ##variable attention
    def __init__(self,  units,dim,score_size,input_size,horizon,alpha,use_GCL=True,seed=123,self_attention=32,k=3,**kwargs):
        super(VA, self).__init__(**kwargs)
        self.units =  units
        self.dim =  dim
        self.score_size =  score_size
        self.input_size =  input_size
        self.horizon =  horizon
        self.alpha = alpha
        self.use_GCL =use_GCL
        self.self_attention =self_attention
        self.k =  k

        tf.random.set_seed(seed)

    def build(self,input_shape):

        #D_C learner
        # self.hX = DynamicEmbeddings(units=self.horizon)

        #use dynamic weighting
        if self.use_GCL:
            self.G = [ [ layers.Dense(self.score_size  ,kernel_regularizer=tf.keras.regularizers.l2(self.alpha)) for i in range(self.input_size)] for h in range(self.horizon)]
            #normalise 
            self.hNorm = [ [ layers.LayerNormalization()   for i in range(self.input_size)] for h in range(self.horizon)]

        #not use dynamic weighting
        if not self.use_GCL:
            self.G = [  layers.Dense(self.score_size  ,kernel_regularizer=tf.keras.regularizers.l2(self.alpha)) for i in range(self.input_size)]
            self.hNorm =   [ layers.LayerNormalization()   for i in range(self.input_size)]


        self.X1 = [ layers.Dense(self.units, activation='elu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(self.alpha)) for i in range(self.input_size)]  # DynamicAttentionLayer(units=units,score_size=score_size,trainable = True)
        self.X2 = [ layers.Dense(self.units,   use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(self.alpha)) for i in range(self.input_size)]  # DynamicAttentionLayer(units=units,score_size=score_size,trainable = True)


        self.context_compressor = [Attention(units=self.self_attention) for h in range(self.horizon)]
        # self.context_compressor = [layers.Dense(units=self.self_attention) for h in range(self.horizon)]
        self.dropout = [layers.Dropout(rate=0.1) for h in range(self.horizon)]


    def call(self, inputs_list):
        if len(inputs_list) > 1:
          inputs  ,horizon_rep =  inputs_list[0],inputs_list[1]
        else:
          inputs    =  inputs_list[0]

        print('inputs',inputs.shape)
        print('horizon_rep',horizon_rep.shape)

        inputs_ = inputs[:,0,:,:]
        print('inputs_',inputs_.shape)
        # print('self.dim ',self.dim )
        contexts = []
        gates = []
        scores = []


        #D_C
        # horizon_rep =  self.hX(  [inputs_] )


        dim_ = 1

        # use Dynamic represntaion
        if self.use_GCL:
            # No normalise 
            # context_ = [ [     layers.Add()( [ self.X2[i]( self.X1[i]( inputs_[:,i,:] ))  , horizon_rep[:,t]] ) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings VA.. ')]# t in range(self.horizon)]
            #normalise 
            # print('horizon_rep ',horizon_rep.shape)
            # print('horizon_rep[:,t], where t =0',horizon_rep[:,0].shape)
            # print('self.X1[i]( inputs_[:,i,:] )), where i =0',self.X1[0]( inputs_[:,0,:] ) .shape)
            # print('self.X2[i]( self.X1[i]( inputs_[:,i,:] )) ,  where i =0' ,self.X2[0]( self.X1[0]( inputs_[:,0,:] )).shape)
            # i =0 
            # print('layers.Add()( [ self.X2[i]( self.X1[i]( inputs_[:,i,:] ))  , horizon_rep[:,t]] ),  where i =0' , layers.Add()( [ self.X2[i]( self.X1[i]( inputs_[:,i,:] ))  , horizon_rep[:,0]] ).shape)
            
            # exit()
            # I changed this line to fix the shape mismatching  issue 
            # context_ = [ [ self.hNorm[t][i] ( layers.Add()( [ self.X2[i]( self.X1[i]( inputs_[:,i,:] ))  , horizon_rep[:,t]] )) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings VA.. ')]# t in range(self.horizon)]
            context_ = [ [ self.hNorm[t][i] ( layers.Add()( [ self.X1[i]( inputs_[:,i,:]) ,self.X2[i](  tf.expand_dims(horizon_rep[:,t], axis=-1)  )] ) )for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings VA.. ')]# t in range(self.horizon)]
            
            

            context =   [ [  self.G[t][i] (context_[t][i]) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'project VA context.. ')]

        # #NOT use Dynamic represntaion
        if not self.use_GCL:
            context_ = [ [ self.hNorm[i] ( self.X2[i]( self.X1[i]( inputs_[:,i,:] ))  ) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings VA.. ')]# t in range(self.horizon)]
            context =   [ [  self.G[i] (context_[t][i]) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'project VA context.. ')]


        score = [ [ tf.nn.relu (context[t][i]) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'find scores VA.. ')]
        print('len(score)',len(score))
        print('len(score[0])',len(score[0]))
        print('score[0][0].shape',score[0][0].shape)
        # print('score',score)
        # print('score',score)
         #reshape
        context = [ [  tf.expand_dims(context[t][i],dim_) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'context reshaping.. ')]
        score = [ [  tf.expand_dims(score[t][i],dim_) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'scores reshaping.. ')]
        score = [ layers.Concatenate(dim_) (score[t]) for t  in  range(self.horizon)]
        context = [ layers.Concatenate(dim_) (context[t]) for t  in  range(self.horizon)]
        print('before compresss => context[t], t=0:',context[0].shape)

        print('len(context)',len(context))
        print('len(context[0])',len(context[0]))
        print('context[0].shape)',context[0].shape)
        print('context[0][0].shape',context[0][0].shape)

        context= [self.dropout[t] (self.context_compressor[t]( [context[t]]  ) ) for t  in tqdm(  range(self.horizon), desc = 'context_compressing.. ')]
        # context= [self.dropout[t] (self.context_compressor[t](  layers.Multiply()( [context[t],score[t] ] )   ) ) for t  in tqdm(  range(self.horizon), desc = 'context_compressing.. ')]
        print('after compresss => context[t], t=0:',context[0].shape)

        print('context after context_compressor')
        print('len(context)',len(context))
        print('len(context[0])',len(context[0]))
        print('context[0][0].shape',context[0][0].shape)

        score = [  tf.expand_dims(score[t],1)  for t  in  range(self.horizon)]
        context = [  tf.expand_dims(context[t],1)  for t  in  range(self.horizon)]

        scores = layers.Concatenate(1) (score)
        contexts = layers.Concatenate(1) (context)

        return contexts, scores



class TA(tf.keras.layers.Layer):
    ##temporal attention
    def __init__(self,  units,dim,score_size,input_size,horizon,alpha,use_GCL=True,seed=123,self_attention=32,k=3,**kwargs):
        super(TA, self).__init__(**kwargs)
        self.units =  units
        self.dim =  dim
        self.score_size =  score_size
        self.input_size =  input_size
        self.horizon =  horizon
        self.alpha = alpha
        self.use_GCL =use_GCL
        self.self_attention =self_attention
        self.k =  k

        tf.random.set_seed(seed)

    def build(self,input_shape):

        #D_C learner

        self.X1 = [layers.LSTM(units=self.units,return_sequences=True,recurrent_dropout=0.1 ,return_state=False,kernel_regularizer=tf.keras.regularizers.l2(self.alpha))  for i in range(self.input_size)]
        self.X = [Attention(units=self.units,return_score=True, horizon=self.horizon   )  for i in range(self.horizon)]

        self.context_compressor = [Attention(units=self.self_attention) for h in range(self.horizon)]
        # self.context_compressor = [layers.Dense(units=self.self_attention) for h in range(self.horizon)]

        self.dropout = [layers.Dropout(rate=0.1) for h in range(self.horizon)]


    def call(self, inputs_list):
        if len(inputs_list)>1:
          inputs  ,horizon_rep =  inputs_list[0],inputs_list[1]
        else:
          inputs   =  inputs_list[0]
        inputs_ = inputs[:,0,:,:]
        print('inputs',inputs_)
        # print('self.dim ',self.dim )
        contexts = []
        gates = []
        scores = []


        #D_C
        # horizon_rep =  self.hX(  [inputs_] )


        dim_ = self.dim
        _1  = [ self.X1[i](inputs_[:,:,i:i+1] )   for i  in  tqdm(range(self.input_size) , desc = 'caclulate embeddings TA (1).. ')]
        print('len _1,  ',len(_1))
        print(_1)
        print('len _1[0],  ',len(_1[0]))
        print('_1[0][0] shape,  ',_1[0][0].shape)
        print('inputs_[:,:,0:1] shape,  ',inputs_[:,:,0:1].shape)
        
        # use Dynamic represntaion
        if self.use_GCL:
            _2  = [[self.X[t]([ _1[i] ,horizon_rep[:,t]]) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings TA.. ')]
            # _2  = [self.X[t]([ _1[i] ,horizon_rep[:,t]]) for i  in  range(self.input_size)]  desc = 'caclulate embeddings TA.. ')]
        print('len _2,  context',len(_2))
        print('len _2[0],  score',len(_2[0]))
        print('_2[0][0] [0] shape,  ',_2[0][0][0].shape)
        print('_2[0][0] [1] shape,  ',_2[0][0][1].shape)

        # #NOT use Dynamic represntaion
        if not self.use_GCL:
            _2  = [[self.X[t]([ _1[i]  ]) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'caclulate embeddings TA.. ')]


        #------------
        score = [[ AmplificationLayer(exponent=0.1) ( _2[t][i][1] ) for i  in  range(self.input_size)] for t in range(self.horizon)]
        context = [[_2[t][i][0] for i  in  range(self.input_size)] for t in range(self.horizon)]
 
        #reshape

        context = [ [  tf.expand_dims(context[t][i],dim_) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'context reshaping.. ')]
        score = [ [  tf.expand_dims(score[t][i],dim_) for i  in  range(self.input_size)] for t in tqdm(range(self.horizon), desc = 'scores reshaping.. ')]
        
        score = [ layers.Concatenate(dim_) (score[t]) for t  in  range(self.horizon)]
        context = [ layers.Concatenate(dim_) (context[t]) for t  in  range(self.horizon)]
        
        
        print('before compresss => context[t], t=0:',context[0].shape)
        context= [self.dropout[t] (self.context_compressor[t]( [context[t]] ) ) for t  in tqdm(  range(self.horizon), desc = 'context_compressing.. ')]
        # context= [self.dropout[t] (self.context_compressor[t](  layers.Multiply()( [context[t],score[t] ] )   ) ) for t  in tqdm(  range(self.horizon), desc = 'context_compressing.. ')]
        print('after compresss => context[t], t=0:',context[0].shape)




        score = [  tf.expand_dims(score[t],1)  for t  in  range(self.horizon)]
        context = [  tf.expand_dims(context[t],1)  for t  in  range(self.horizon)]

        scores = layers.Concatenate(1) (score)
        contexts = layers.Concatenate(1) (context)
        print('contexts',contexts)
        print('scores',scores)

        return contexts, scores


class MDA(tf.keras.layers.Layer):
    '''
    Multidimensional Dynamical Attention (MDA) layer
    '''
    def __init__(self, no_variables,timesteps,horizon,shared_ta_units=3,shared_va_units=3,alpha=0.01, units=1,use_TA=True,use_VA=True,use_GCL=True,use_context=True,approach='sequential', static_attention=False,seed=123,self_attention=32,k=3,**kwargs):
        super(MDA, self).__init__(**kwargs)
        self.units = units
        self.shared_ta_units = shared_ta_units
        self.shared_va_units = shared_va_units
        self.use_TA=use_TA
        self.use_VA=use_VA
        self.type= approach #'sequential' #parallel , 'sequential
        self.novaribles = no_variables
        self.timesteps = timesteps
        self.horizon = horizon
        self.use_context= use_context
        self.alpha = alpha
        self.static_attention = static_attention
        self.use_GCL=use_GCL
        self.seed = seed
        self.self_attention =self_attention
        self.k = k
        tf.random.set_seed(seed)

        reset_seed(seed)


    def build(self,input_shape):
        shared_DRL = False
        if shared_DRL:
            horizon_rep  = DynamicEmbeddings(units=self.horizon)


        if self.use_VA :
            if shared_DRL:
                self.hX_v = horizon_rep #DynamicEmbeddings(units=self.horizon)
            else:
                self.hX_v =  DynamicEmbeddings(units=self.horizon)
            self.va_unit = VA( units=self.shared_va_units,dim=2,alpha=self.alpha,score_size=self.novaribles,input_size=self.timesteps,horizon=self.horizon, use_GCL=self.use_GCL ,seed = self.seed,self_attention=self.self_attention,k = self.k )
        if self.use_TA :
            self.ta_unit = TA( units=self.shared_ta_units,dim=-1,alpha=self.alpha, score_size=self.timesteps,input_size=self.novaribles,horizon=self.horizon, use_GCL=self.use_GCL,seed = self.seed ,self_attention=self.self_attention )
            if shared_DRL:
                self.hX_t = horizon_rep #DynamicEmbeddings(units=self.horizon)
            else:
                self.hX_t = DynamicEmbeddings(units=self.horizon)



    def call(self, inputs ):  # Defines the computation from inputs to outputs.
        alphas =[]
        contexts =[]

        inputs_ = inputs[:,0,:,:]

        #uae horizon rep
        if self.use_GCL:
            if self.use_VA :
                horizon_rep_v =  self.hX_v(  [inputs_] )

            if self.use_TA :
                horizon_rep_t =  self.hX_t(  [inputs_] )


        if self.use_VA:
            if self.use_GCL:
                print("horizon_rep_v")
                print(horizon_rep_v.shape)
                context_v ,  score_v =   self.va_unit( [inputs, horizon_rep_v] )


            if not self.use_GCL:
                context_v ,  score_v =   self.va_unit( [inputs] )

            alphas.append(score_v)
            contexts.append(context_v)
        
        print("\n\n----------- VA done -----------\n\n")
    
        if self.type== 'parallel'  :
            selected_seq = inputs
        elif self.type== 'sequential'  :
            selected_seq = layers.Multiply()([inputs,alphas[0]])


        if self.use_TA:
            if self.use_GCL:
                context_t ,  score_t =   self.ta_unit([selected_seq,  horizon_rep_t])

            if not self.use_GCL:
                context_t ,  score_t =   self.ta_unit([selected_seq ])
            alphas.append(score_t)
            contexts.append(context_t)
        print('context_v',context_v.shape)
        print('context_t',context_t.shape)
        print('score_v',score_v.shape)
        print('score_t',score_t.shape)
        print ('go to agu')
        # exit()

        return self.AGU(alphas,contexts,selected_seq )


    def AGU (self, alphas,contexts,inputs  ):
        self.scores_length = len(alphas)
        
        if len(alphas) > 1:
            score = layers.Multiply ()(alphas)


            if self.use_context:
                print('use_context')
                context_times_score = layers.Multiply()([inputs,score])
                context_times_score = layers.Reshape((context_times_score.shape[1],context_times_score.shape[2]*context_times_score.shape[3]))(context_times_score)
                print('context_times_score',context_times_score)
                if self.use_VA :
                  context_v = contexts[0]
                #   print('context_v',context_v)
                if self.use_TA :
                  if len(contexts)>1:
                    id = 1
                  else:
                    id =0
                  context_t = contexts[id]

                # combined_context = [context_times_score,context_v,context_t]
                combined_context = [context_v,context_t]
                contexts = layers.Concatenate(-1)(combined_context)
                print('contexts',contexts)


            else:
                print('No use_context')

                contexts =  layers.Multiply()([inputs,score])


        else :
            score = alphas [0]
            # contexts =contexts[0]
            contexts = layers.Multiply()([inputs,score])
            contexts = layers.Reshape((contexts.shape[1],contexts.shape[2]*contexts.shape[3]))(contexts)

        print('contexts',contexts)
        # exit()

        return score , contexts


class MAFS_extend(object):

    def __init__(self ,params ):
        print('MAFS_extend')
        # tf.keras.backend.clear_session()
        self.params=params
        tf.random.set_seed(self.params['seed'])




    def build_model(self):

        input_seq = layers.Input(( self.params['time_steps'] , self.params['no_varibles']))
        # print('input_seq',input_seq.shape)

        # Lists to store outputs
        outputs = list()
        scores_gates = list()

        # new_represntaion = input_seq
        ## add dimension to the input to represnt the prediciton horizon
        new_represntaion = tf.expand_dims(input_seq,1)
        new_represntaion = layers.Concatenate(1)([new_represntaion for _ in range(self.params['horizon'] )])
        # print('new_represntaion',new_represntaion.shape)
        
        scores,contexts = MDA( units=self.params['units'],\
                                    use_TA=self.params['use_TA'],\
                                    use_VA=self.params['use_VA'], \
                                    use_GCL=self.params['use_GCL'], \
                                    no_variables=self.params['no_varibles'] ,\
                                    timesteps=self.params['time_steps'],\
                                    horizon=self.params['horizon'],\
                                    use_context=self.params['use_context'],\
                                    # shared_attention = self.params['shared_attention']  ,\
                                    shared_ta_units = self.params['shared_units_ta'],\
                                    shared_va_units= self.params['shared_units_va'],\
                                    alpha= self.params['alpha'],\
                                    approach = self.params['type'],\
                                  static_attention=self.params['static_attention'],
                                  seed =self.params['seed'],\
                                  self_attention=self.params['self_attention'],
                                  k = self.params['k']
                                    )(new_represntaion)


        print('>> contexts',contexts)

        outputs = layers.TimeDistributed(layers.Dense( self.params['learning_task_hidden'], activation = 'tanh'))(contexts)
        outputs =  layers.Dropout( self.params['dropout_rate'] )(outputs)
        outputs = layers.TimeDistributed(layers.Dense( 1 ))(outputs)
        outputs = layers.Reshape( (self.params['horizon'],))(outputs)


        scores_gates = [outputs, scores]


        # model = Model([input_seq, hs_in,cs_in], outputs)
        model = Model(input_seq, outputs)

        model = compile(model = model, params = self.params)

        weights_model = Model(input_seq, scores_gates)##,A_V_gate , A_T_gate] )
        # print(model.summary())







        return model, weights_model


def learning_task(horizon,h_s, x, activation='None', name='',use_timedistributed=True):
    if use_timedistributed:
        # outputs = tf.keras.layers.RepeatVector(  horizon) ( x)
        outputs = tf.keras.layers.TimeDistributed(  layers.Dense(h_s,activation=activation )) ( x)
        outputs = tf.keras.layers.TimeDistributed(layers.Dense(1 )) ( outputs)
    else:

        outputs =layers.Dense(h_s,activation=activation ) ( x)
        outputs =layers.Dense(horizon )( outputs)
    return outputs


