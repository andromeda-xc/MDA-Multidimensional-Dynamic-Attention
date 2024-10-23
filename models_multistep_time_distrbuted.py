

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
    # adam = tf.keras.optimizers.Nadam(  lr=float(params['learning_rate']))#, clipnorm=float(params['max_gradient_norm']) )


    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=float(params['learning_rate']) )#,weight_decay = True)
    model.compile( loss=loss, optimizer=optimizer, metrics=['mae']) #'mse')
    # print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True,  expand_nested=True, show_layer_names=True  )
    return model



class NN(object):

  def __init__(self ,params ):
      print('NN')

      tf.keras.backend.clear_session()

      tf.random.set_seed(params['seed'])

      self.params=params
  def build_model(self):
      input_seq = layers.Input(( self.params['time_steps'] ,  self.params['no_varibles'] ) )
      generated_seq_f = layers.Dense(self.params['h_s'], activation='relu') (input_seq)
      generated_seq_f = layers.Dropout(self.params['dropout_rate'])(generated_seq_f)
      generated_seq_f = layers.Dense(self.params['h_s'])(generated_seq_f)
      generated_seq_f = layers.Dropout(self.params['dropout_rate'])(generated_seq_f)

      generated_seq_f = layers.Flatten()(generated_seq_f)
      # generated_seq_f = layers.RepeatVector(self.params['horizon'])(generated_seq_f)
      # generated_seq_f = layers.TimeDistributed(layers.Dense(self.params['horizon'], kernel_initializer=tf.initializers.zeros()))(generated_seq_f)
      outputs = layers.Dense(self.params['horizon'] ) (generated_seq_f)

      #  outputs =  learning_task(horizon=1 ,use_timedistributed = True, \
      #     h_s=self.params['h_s'], x = generated_seq_f, activation=None)

      # outputs = layers.Reshape( (self.params['horizon'],))(outputs)

      # if use_rf:
      #     model = keras.models.Model([inputs,rf_weights], outputs)
      # else:
      model = Model(input_seq, outputs)
      model = compile(model = model, params = self.params)

      return model



class CNN(object):
    def __init__(self ,params ):
        print('CNN')
        tf.keras.backend.clear_session()
        tf.random.set_seed(params['seed'])

        self.params=params
    def build_model(self):
        input_seq = layers.Input(( self.params['time_steps'] ,  self.params['no_varibles'] ) )
        generated_seq_f = layers.Conv1D(self.params['h_s'], kernel_size= 3,padding='same',activation='tanh') (input_seq)
        for i  in range(1,self.params['no_enc_layer'] ):
            generated_seq_f = layers.Conv1D(self.params['h_s'], kernel_size= 3 ) (generated_seq_f)
            generated_seq_f = layers.Dropout(self.params['dropout_rate'])(generated_seq_f)
        generated_seq_f = layers.BatchNormalization()(generated_seq_f)
        generated_seq_f = layers.Dense(self.params['h_s'] ,activation='relu')(generated_seq_f)

        generated_seq_f = layers.Flatten()(generated_seq_f)
        # generated_seq_f = layers.RepeatVector(self.params['horizon'])(generated_seq_f)
        # generated_seq_f = layers.TimeDistributed(layers.Dense(self.params['horizon'], kernel_initializer=tf.initializers.zeros()))(generated_seq_f)
        generated_seq_f = layers.Dense(self.params['h_s'] ) (generated_seq_f)
        outputs = layers.Dense(self.params['horizon'] , kernel_regularizer= tf.keras.regularizers.L2(l2=0.001)) (generated_seq_f)

        #  outputs =  learning_task(horizon=1 ,use_timedistributed = True, \
        #     h_s=self.params['h_s'], x = generated_seq_f, activation=None)

        outputs = layers.Reshape( (self.params['horizon'],))(outputs)

        # if use_rf:
        #     model = keras.models.Model([inputs,rf_weights], outputs)
        # else:
        model = Model(input_seq, outputs)
        model = compile(model = model, params = self.params)

        return model


class CNNLSTM(object):
    def __init__(self ,params ):
        print('CNNLSTM')

        tf.keras.backend.clear_session()
        tf.random.set_seed(params['seed'])

        self.params=params
    def build_model(self):

        input_seq = layers.Input(( self.params['time_steps'],  self.params['no_varibles']))
        initializer = tf.keras.initializers.GlorotNormal()
        n_hidden = self.params['h_s']



        conv_enncoder = layers.Conv1D(30, 3,padding='same',activation='relu')(input_seq)
        # conv_enncoder = layers.MaxPool1D(2)(conv_enncoder)
        conv_enncoder = layers.BatchNormalization()(conv_enncoder)
        conv_enncoder = layers.Dropout( self.params['dropout_rate'])(conv_enncoder)

        encoder_stack_h, encoder_last_h, encoder_last_c = layers.LSTM(
            n_hidden,  dropout=0.2, recurrent_dropout=0.2,
            return_state=True, return_sequences=True,kernel_initializer=initializer )(conv_enncoder)

        encoder_last_h = layers.BatchNormalization()(encoder_last_h)
        print(encoder_last_h)
        encoder_last_c = layers.BatchNormalization()(encoder_last_c)
        print(encoder_last_c)

        decoder_input = layers.RepeatVector(self.params['horizon'])(encoder_last_h)
        print(decoder_input)

        decoder_stack_h = layers.LSTM(n_hidden,   dropout=0.2, recurrent_dropout=0.2,
                      return_state=False, return_sequences=True,kernel_initializer=initializer )(
            decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        print(decoder_stack_h)

        if self.params['model_name'] == 'atCNNLSTM' :
          attention = layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
          attention = layers.Activation('softmax')(attention)
          context = layers.dot([attention, encoder_stack_h], axes=[2,1])
          context = layers.BatchNormalization()(context)
          decoder_combined_context = layers.Concatenate()([context, decoder_stack_h])

        else:

          context = layers.BatchNormalization()(decoder_stack_h)
          decoder_combined_context =context

        outputs = layers.TimeDistributed(layers.Dense(1))(decoder_combined_context)



        #  outputs =  learning_task(horizon=1 ,use_timedistributed = True, \
        #     h_s=self.params['h_s'], x = generated_seq_f, activation=None)

        outputs = layers.Reshape( (self.params['horizon'],))(outputs)

        # if use_rf:
        #     model = keras.models.Model([inputs,rf_weights], outputs)
        # else:
        model = Model(input_seq, outputs)
        model = compile(model = model, params = self.params)

        return model



class STAM(object):
  def __init__(self ,params ):
    self.params = params
    tf.keras.backend.clear_session()
    tf.random.set_seed(params['seed'])

    # Model
    self.t_repeator = layers.RepeatVector(Tx)
    self.t_densor = layers.Dense(1, activation = "relu")

    self.s_repeator = layers.RepeatVector(inp_var)
    self.s_densor_1 = layers.Dense(self.params['h_s'], activation = "relu")
    self.s_densor_2 = layers.Dense(1, activation = "relu")   # for attention weights

    self.concatenator = layers.Concatenate(axis=-1)

    self.activator = layers.Activation(self.softMaxLayer)
    self.dotor = layers.Dot(axes = 1)

    self.s_decoder_lstm = layers.LSTM(self.params['h_s'], return_state = True)
    self.t_decoder_lstm = layers.LSTM(self.params['h_s'], return_state = True)

    self.flatten = layers.Flatten()
  # Softmax
  def softMaxLayer(x):
      return tf.keras.activations.softmax(x, axis=1)   # Use axis = 1 for attention


  # Temporal Attention
  def temporal_attention(self,a, s_prev):

      # s_prev: previous hidden state of decoder (n_samples, 16)
      # a: Sequence of encoder hidden states (n_sample, 10, 16)
      s_prev = self.t_repeator(s_prev)  # (n_samples, 10, 16)
      concat = self.concatenator([a, s_prev])   # (n_samples, 10, 32)
      e_temporal = self.t_densor(concat)  # (n_samples, 10, 1)
      alphas = self.activator(e_temporal)    # (n_samples, 10, 1)
      t_context = self.dotor([alphas, a])    # (n_samples, 1, 16)

      return t_context, alphas, e_temporal

  # Spatial Attention
  def spatial_attention(self,v, s_prev):

      # s_prev: previous hidden state of decoder (n_samples, 16)
      # v: variable vectors (n_samples, 8, 10): (n_samples, inp_var, Tx)
      s_fc = self.s_densor_1(v)   # D .. spatial embeddings ..   # (n_samples, 8, 16)

      s_prev = self.s_repeator(s_prev)  # (n_samples, 8, 16)
      concat = self.concatenator([s_fc, s_prev])    # (n_samples, 8, 32)
      e_spatial = self.s_densor_2(concat)  # (n_samples, 8, 1)
      betas = self.activator(e_spatial) # (n_samples, 8, 1)
      s_context = self.dotor([betas, s_fc])  # (n_samples, 1, 16)

      return s_context, betas, e_spatial


  def build_model(self):

      encoder_input = layers.Input(shape = (self.params['time_steps'], self.params['no_varibles']),name='encoder_input')   # (None, 10, 8)
      spatial_input = layers.Input(shape = (self.params['no_varibles'], self.params['time_steps']), name='spatial_input' )    # (None, 8, 10)


      # Initialize
      s0 = layers.Input(shape=(self.params['h_s'],), name= 's0')  # Initialize hidden state for decoder   (None, 16)
      c0 = layers . Input(shape=(self.params['h_s'],), name= 'c0')  # Initialize cell state for decoder     (None, 16)
      yhat0 = layers.Input(shape=(1, ), name='yhat0')  # Initialize prev pred y   (None, 1)
      ts, tc = s0, c0  # Temporal LSTM
      ss, sc = s0, c0  # Spatial LSTM   spatial hidden states and cell states  , في البدايه هي نفس الانبوت
      yhat = yhat0  # For regre

      # Lists to store outputs
      outputs = list()
      alphas_betas_list = list()


      # Encoder LSTM, Pre-attention
      lstm_1, state_h, state_c = layers.LSTM(self.params['h_s'], return_state=True, return_sequences=True)(encoder_input)
      lstm_1 = layers.Dropout (self.params['dropout_rate'])(lstm_1)     # (None, 10, 16)

      lstm_2, state_h, state_c = layers.LSTM(self.params['h_s'], return_state=True, return_sequences=True)(lstm_1)
      lstm_2 = layers.Dropout (self.params['dropout_rate'])(lstm_2)     # (None, 10, 16)


      # Decode for Ty steps

      for t in range(self.params['horizon']):

          # Temporal Attention
          t_context, alphas, e_temporal = self.temporal_attention (lstm_2, ts)  # (None, 1, 16)

          t_context = layers.Dense (self.params['encode_dim'], activation = "relu")(t_context)  # (None, 1, 4)
          t_context = self.flatten(t_context)  # (None, 4)
          t_context = self.concatenator([t_context, yhat])   # (None, 5)
          t_context = layers.Reshape((1, self.params['encode_dim'] + 1), name='t_context_'+str(t) )(t_context)   # (None, 1, 5)

          # Spatial Attention
          s_context, betas, e_spatial = self.spatial_attention (spatial_input, ss)   # (None, 1, 16)

          s_context = layers.Dense (self.params['encode_dim'], activation = "relu")(s_context)
          s_context = self.flatten(s_context)  # (None, 4)
          s_context = self.concatenator([s_context, yhat])   # (None, 5)
          s_context = layers.Reshape((1, self.params['encode_dim'] + 1), name='s_context'+str(t) )(s_context)   # (None, 1, 5)

          # T Decoder LSTM
          ts, _, tc = self.t_decoder_lstm(t_context, initial_state=[ts, tc])
          ts = layers.Dropout (self.params['dropout_rate'])(ts)   # (None, 16)

          # S Decoder LSTM
          ss, _, sc = self.s_decoder_lstm(s_context, initial_state=[ss, sc])
          ss = layers.Dropout (self.params['dropout_rate'])(ss)   # (None, 16)

          context = self.concatenator([ts, ss])   # (None, 32)

          # FC Layer
          #context = Dense (h_s, activation = "relu")(context)   # (None, 16)



          # FC Layer
          yhat = layers.Dense (1, activation = "linear", name='out_'+str(t))(context)

          # Append lists
          outputs.append(yhat)

          # Append lists
          alphas_betas_list.append(yhat)
          alphas_betas_list.append(alphas)
          alphas_betas_list.append(betas)



      pred_model = Model([encoder_input, spatial_input, s0, c0, yhat0], outputs)   # Prediction Model
      prob_model = Model([encoder_input, spatial_input, s0, c0, yhat0], alphas_betas_list)    # Weights Model
      pred_model = compile(model = pred_model, params = self.params)

      return pred_model, prob_model


  # Softmax
  def softMaxLayer(self, x):
      return tf.keras.activations.softmax(x, axis=1)   # Use axis = 1 for attention

  # Temporal Attention
  def temporal_attention(self , a, s_prev):

      # s_prev: previous hidden state of decoder (n_samples, 16)
      # a: Sequence of encoder hidden states (n_sample, 10, 16)
      s_prev = self.t_repeator(s_prev)  # (n_samples, 10, 16)
      concat = self.concatenator([a, s_prev])   # (n_samples, 10, 32)
      e_temporal = self.t_densor(concat)  # (n_samples, 10, 1)
      alphas = self.activator(e_temporal)    # (n_samples, 10, 1)
      t_context = self.dotor([alphas, a])    # (n_samples, 1, 16)

      return t_context, alphas, e_temporal


  # Spatial Attention
  def spatial_attention(self, v, s_prev):

      # s_prev: previous hidden state of decoder (n_samples, 16)
      # v: variable vectors (n_samples, 8, 10): (n_samples, inp_var, Tx)
      s_fc = self.s_densor_1(v)   # D .. spatial embeddings ..   # (n_samples, 8, 16)

      s_prev = self.s_repeator(s_prev)  # (n_samples, 8, 16)
      concat = self.concatenator([s_fc, s_prev])    # (n_samples, 8, 32)
      e_spatial = self.s_densor_2(concat)  # (n_samples, 8, 1)
      betas = self.activator(e_spatial) # (n_samples, 8, 1)
      s_context = self.dotor([betas, s_fc])  # (n_samples, 1, 16)

      return s_context, betas, e_spatial




class TFT(object):

    global Dropout , Activation,Lambda,Add, stack,LayerNorm,Dense
    Dropout = tf.keras.layers.Dropout
    Activation = tf.keras.layers.Activation
    Lambda = tf.keras.layers.Lambda
    Add = tf.keras.layers.Add
    stack = tf.keras.backend.stack
    LayerNorm = tf.keras.layers.LayerNormalization
    Dense = tf.keras.layers.Dense

    def __init__(self ,params ):
        self.params  = params

        tf.keras.backend.clear_session()
        tf.random.set_seed(params['seed'])
        self.state_h = layers.Input(shape=(params['h_s'] ) )
        self.state_c = layers.Input(shape=(params['h_s']))


    def build_model(self):
        input_data = layers.Input( shape = (self.params['time_steps'], self.params['no_varibles']),name='input')
        embedding= tf.keras.backend.stack(
            [
            self.convert_real_to_embedding(input_data[:, :,i:i + 1])
            for i in range( input_data.shape[2])
            ],  axis=-1)

            # [
            # self.convert_real_to_embedding(input_data[:, i:i+1,j:j+ 1])
            # for i , j in zip(range( input_data.shape[1]), range( input_data.shape[2]))
            # ],  axis=-1)


        # print('embedding',embedding)

        self.convert_real_to_embedding(input_data)
        # print(embedding)
        _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()
        flatten = K.reshape(embedding,
                          [-1, time_steps, embedding_dim * num_inputs])

        # Variable selection weights
        mlp_outputs, static_gate = self.gated_residual_network(
            flatten,
            self.params['h_s'],
            output_size=num_inputs,
            dropout_rate=self.params['dropout_rate'],
            use_time_distributed=True,
            additional_context=None,
            return_gate=True)


        sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=2)



        # Non-linear Processing & weight application
        trans_emb_list = []
        for i in range(num_inputs):
            grn_output = self.gated_residual_network(
                embedding[Ellipsis, i], ##to do
                self.params['h_s'],
                dropout_rate=self.params['dropout_rate'],
                use_time_distributed=True)
            trans_emb_list.append(grn_output)

        transformed_embedding = stack(trans_emb_list, axis=-1)

        combined = tf.keras.layers.Multiply()(
            [sparse_weights, transformed_embedding])
        temporal_ctx = K.sum(combined, axis=-1)

        historical_features, historical_flags  = temporal_ctx  , sparse_weights
        # print('historical_flags:',historical_flags)
        # s0 = layers.Input(shape=(h_s,), name= 's0')  # Initialize hidden state for decoder   (None, 16)
        # c0 = layers.Input(shape=(h_s,), name= 'c0')

        state_h = self.state_h
        state_c = self.state_c
        history_lstm, state_h, state_c \
            = self.get_lstm(return_state=True)(historical_features,
                                        initial_state=[state_h,
                                                      state_c  ])

        input_embeddings =historical_features

        lstm_layer, _ = self.apply_gating_layer(
            history_lstm, self.params['h_s'] , self.params['dropout_rate']  , activation=None)

        temporal_feature_layer = self.add_and_norm([lstm_layer, input_embeddings])
        # print('temporal_feature_layer',temporal_feature_layer)
        # print('historical_flags',historical_flags[:,:,0,:])



        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.params['num_heads'], self.params['h_s'], dropout=self.params['dropout_rate'])

        # print('temporal_feature_layer',temporal_feature_layer.shape)
        mask = self.get_decoder_mask(temporal_feature_layer)
        x, self_att \
            = self_attn_layer(temporal_feature_layer, temporal_feature_layer, temporal_feature_layer,
                            mask=mask)
        # print('mask',mask.shape)
        # print('x',x.shape)
        # print('self_att',self_att.shape)



        x, _ = self.apply_gating_layer(
            x,
            self.params['h_s'],
            dropout_rate=self.params['dropout_rate'],
            activation=None)

        x = self.add_and_norm([x, temporal_feature_layer])


        # Nonlinear processing on outputs
        decoder = self.gated_residual_network(
            x,
            self.params['h_s'],
            dropout_rate=self.params['dropout_rate'],
            use_time_distributed=True)

        # Final skip connection
        decoder, _ = self.apply_gating_layer(
            decoder, self.params['h_s'] , activation=None)
        out = self. add_and_norm([decoder, temporal_feature_layer])
        out = layers.Flatten() (out)
        out = learning_task(horizon =self.params['horizon']  ,  x = out, use_timedistributed=False,  activation=None, h_s=self.params['h_s'] )
        out = layers.Reshape( (self.params['horizon'],))(out)


        model = Model(inputs = [input_data,self.state_h ,self.state_c]  , outputs = out)
        weights_model =  Model(inputs = [input_data,self.state_h ,self.state_c] , outputs = [out , historical_flags[:,:,0,:],self_att[0,:,:,:]])


        model = compile(model=model , params=self.params)

        return model, weights_model

    def convert_real_to_embedding(self, x):
      """Applies linear transformation for time-varying inputs."""
      return tf.keras.layers.TimeDistributed(
          tf.keras.layers.Dense(self.params['h_s']))( x)

    # Layer utility functions.
    def linear_layer(self,size,
                    activation=None,
                    use_time_distributed=False,
                    use_bias=True):
        """Returns simple Keras linear layer.

        Args:
            size: Output size
            activation: Activation function to apply if required
            use_time_distributed: Whether to apply layer across time
            use_bias: Whether bias should be included in layer
        """
        linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        return linear


    def apply_gating_layer(self,x,
                        hidden_layer_size,
                        dropout_rate=None,
                        use_time_distributed=True,
                        activation=None):
        """Applies a Gated Linear Unit (GLU) to an input.

        Args:
            x: Input to gating layer
            hidden_layer_size: Dimension of GLU
            dropout_rate: Dropout rate to apply if any
            use_time_distributed: Whether to apply across time
            activation: Activation function to apply to the linear feature transform if
            necessary

        Returns:
            Tuple of tensors for: (GLU output, gate)
        """

        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if use_time_distributed:
            activation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
                    x)
            gated_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
                    x)
        else:
            activation_layer = tf.keras.layers.Dense(
                hidden_layer_size, activation=activation)(
                    x)
            gated_layer = tf.keras.layers.Dense(
                hidden_layer_size, activation='sigmoid')(
                    x)

        return tf.keras.layers.Multiply()([activation_layer,
                                            gated_layer]), gated_layer

    def add_and_norm(self, x_list):
        """Applies skip connection followed by layer normalisation.

        Args:
            x_list: List of inputs to sum for skip connection

        Returns:
            Tensor output from layer.
        """
        tmp = Add()(x_list)
        tmp = LayerNorm()(tmp)
        return tmp

    def gated_residual_network(self, x,
                            hidden_layer_size,
                            output_size=None,
                            dropout_rate=None,
                            use_time_distributed=True,
                            additional_context=None,
                            return_gate=False):
        """Applies the gated residual network (GRN) as defined in paper.

        Args:
            x: Network inputs
            hidden_layer_size: Internal state size
            output_size: Size of output layer
            dropout_rate: Dropout rate if dropout is applied
            use_time_distributed: Whether to apply network across time dimension
            additional_context: Additional context vector to use if relevant
            return_gate: Whether to return GLU gate for diagnostic purposes

        Returns:
            Tuple of tensors for: (GRN output, GLU gate)
        """

        # Setup skip connection
        if output_size is None:
            output_size = hidden_layer_size
            skip = x
        else:
            linear = Dense(output_size)
            if use_time_distributed:
                linear = tf.keras.layers.TimeDistributed(linear)
            skip = linear(x)

        # Apply feedforward network
        hidden = self.linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed)(
                x)
        if additional_context is not None:
            hidden = hidden + self.linear_layer(
                hidden_layer_size,
                activation=None,
                use_time_distributed=use_time_distributed,
                use_bias=False)(
                    additional_context)
        hidden = tf.keras.layers.Activation('elu')(hidden)
        hidden = self.linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed)(
                hidden)

        gating_layer, gate = self.apply_gating_layer(
            hidden,
            output_size,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
            activation=None)

        if return_gate:
            return self.add_and_norm([skip, gating_layer]), gate
        else:
            return self.add_and_norm([skip, gating_layer])

    # LSTM layer
    def get_lstm(self,return_state):
      """Returns LSTM cell initialized with default parameters."""
      if self.params['use_cudnn']:
        lstm = tf.compat.v1.keras.layers.CuDNNLSTM(
            self.params['h_s'],
            return_sequences=True,
            return_state=return_state,
            stateful=False,
        )
      else:
        lstm = tf.keras.layers.LSTM(
            self.params['h_s'],
            return_sequences=True,
            return_state=return_state,
            stateful=False,
            # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
            # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True)
      return lstm


    # Attention Components.
    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = tf.shape(self_attn_inputs)[1]
        bs = tf.shape(self_attn_inputs)[:1]
        mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
        return mask



class ScaledDotProductAttention():
    """Defines scaled dot product attention layer.

    Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
        softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
        q: Queries
        k: Keys
        v: Values
        mask: Masking if required -- sets softmax to very large value

        Returns:
        Tuple of (layer outputs, attention weights)
        """
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)(
            [q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(
                mask)  # setting to infinity
        attn = Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention():



    """Defines interpretable multi-head attention layer.

    Attributes:
        n_head: Number of heads
        d_k: Key/query dimensionality per head
        d_v: Value dimensionality
        dropout: Dropout rate to apply
        qs_layers: List of queries across heads
        ks_layers: List of keys across heads
        vs_layers: List of values across heads
        attention: Scaled dot product attention layer
        w_o: Output weight matrix to project internal state to the original TFT
        state size
    """

    def __init__(self, n_head, d_model, dropout):
        """Initialises layer.

        Args:
        n_head: Number of heads
        d_model: TFT state dimensionality
        dropout: Dropout discard rate
        """

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interp
        vs_layer = Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False , name = 'head_'+str(_)))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention =  ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

        Using T to denote the number of time steps fed into the transformer.

        Args:
        q: Query tensor of shape=(?, T, d_model)
        k: Key of shape=(?, T, d_model)
        v: Values of shape=(?, T, d_model)
        mask: Masking if required with shape=(?, T, T)

        Returns:
        Tuple of (layer outputs, attention weights)
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        outputs = K.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = Dropout(self.dropout)(outputs)  # output dropout

        return outputs, attn



class MAFS(object):
    def __init__(self ,params ):
        print('MAFS')
        tf.keras.backend.clear_session()
        tf.random.set_seed(params['seed'])
        self.params=params
        self.score_based = 'multiply'
    def build_model(self):

        input_seq = layers.Input(( self.params['time_steps'] ,  self.params['no_varibles']))



        A_V =[]
        variables_num = self.params['no_varibles']
        time_steps =  self.params['time_steps']
        for t in range(time_steps):
            inp_slice = input_seq[:,t,:]
            # inp_slice = tf.expand_dims(inp_slice,-1)
            # print('inp_slice',inp_slice)
            layer =  layers.Dense(self.params['h_s'],   activation ='tanh',kernel_initializer =tf.keras.initializers.Ones()  )  (inp_slice)
            # layer =  layers.Dense(32, activation='tanh',kernel_regularizer=tf.keras.regularizers.L2(l2=0.001 )  )  (layer)
            # print('layer',layer)

            layer =  layers.Dense(variables_num, activation='softmax',kernel_initializer =tf.keras.initializers.Ones()  )  (layer)
            # layer =  layers.Dense(1, activation='softmax',kernel_regularizer=tf.keras.regularizers.L2(l2=0.001 ) , name = 'softA_V_'+str(t) )  (layer)
            # print('layer softmax',layer)

            layer =  tf.expand_dims(layer,axis= 1 )
            # layer =  layers.Lambda(lambda x: tf.math.truediv(x,  tf.reduce_max(x)), name='scale'+'A_V_'+str(t) )(layer)
            # layer = tf.nn.softmax(layer, axis=-1)
            # layer =  layers.Lambda(lambda a: tf.where( tf.equal(a, tf.reduce_max(a)), a, tf.zeros_like(a))     , name='scale'+'A_V_'+str(t) )(layer)
            # layer =  layers.Lambda(lambda a: tf.where( tf.equal(a, tf.reduce_max(a ,axis=-1,keepdims=True)), a, tf.constant(value=0.01 ))     , name='scale'+'A_V_'+str(t) )(layer)
            # layer =  layers.Lambda(lambda a: tf.where( tf.equal(a, tf.reduce_max(a ,axis=-1,keepdims=True)), a, tf.zeros_like(a))     , name='scale'+'A_V_'+str(t) )(layer)
            A_V.append(layer )
        A_V = layers.Concatenate(name='A_V',axis=1)(A_V)
        # A_V =  K.permute_dimensions(A_V, (0,2,1))
        # print('A_V',A_V)

        A_T =[]
        for v in range(variables_num):
            inp_slice = input_seq[:,:,v]
            # inp_slice = tf.expand_dims(inp_slice,-1)
            # print('inp_slice',inp_slice)
            layer =  layers.Dense(self.params['h_s'] , activation ='tanh',kernel_initializer = tf.keras.initializers.Ones() )  (inp_slice)

            layer = layers.Dense(time_steps,  activation='softmax',kernel_initializer =tf.keras.initializers.Ones()   )  (layer)


            layer =  tf.expand_dims(layer, -1)
            A_T.append(layer )

        A_T = layers.Concatenate(name='A_T',axis= -1)(A_T)


        if self.score_based =='learn':
            score = layers.Concatenate(name='score1',axis=-1)([A_T, A_V])
            # print('score',score.shape)
            # generated_seq_f = layers.Conv2D(1,3,padding='same' ,kernel_initializer =keras.initializers.Ones() ,kernel_regularizer=keras.regularizers.L2(l2=0.001) ,kernel_constraint = keras.constraints.MinMaxNorm() ,use_bias=False)(generated_seq_f)
            score = layers.Dense(1, activation='relu'   ,kernel_constraint = tf.keras.constraints.non_neg() ,use_bias=True)(score)
            print('score',score.shape)
            score = layers.Reshape(target_shape=(generated_seq_f.shape[1],generated_seq_f.shape[2]), name='score' )(score)
            print('score',score.shape)



        elif  self.score_based =='multiply':
            score = layers.Multiply(name='score')([A_T, A_V])
        elif  self.score_based =='add':
            score = layers.Add(name='score')([A_T, A_V])
        else:
            print('Error.. ! chose proper method to combine scores ')


        # if  feature_selction_pre_process or use_all_inputs:
        #     generated_seq_f =input_seq


        # else:
        generated_seq_f = layers.Multiply(name='score_2')([score,input_seq])



        # generated_seq_f = layers.Dense( 100, activation='relu') (generated_seq_f)
        # generated_seq_f = layers.Dense(50 , activation='relu') (generated_seq_f)
        # generated_seq_f = layers.Flatten()(generated_seq_f)
        # outputs = layers.Dense(self.params['horizon']) (generated_seq_f)
        generated_seq_f = layers.Flatten()(generated_seq_f)
        outputs =  learning_task(horizon=self.params['horizon'] , \
            h_s=self.params['h_s'], x = generated_seq_f, activation='relu' ,use_timedistributed=False)


        outputs = layers.Reshape( (self.params['horizon'],))(outputs)

        # if use_rf:
        #     model = keras.models.Model([inputs,rf_weights], outputs)
        # else:
        model = Model(input_seq, outputs)
        weights_model = Model(input_seq, [outputs, score] )
        model = compile(model = model, params = self.params)

        return model, weights_model









class RidgeRegressionWithGLU(tf.keras.layers.Layer):
    def __init__(self, units, alpha=0.01, **kwargs):
        super(RidgeRegressionWithGLU, self).__init__(**kwargs)
        self.units = units
        self.alpha = alpha

    def build(self, batch_input_shape):
        self.w = self.add_weight(
            shape=(batch_input_shape[-1], self.units),
            initializer='random_normal',
            regularizer=tf.keras.regularizers.l2(self.alpha),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(RidgeRegressionWithGLU, self).build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return z * tf.nn.sigmoid(z)



class GLULayer(tf.keras.layers.Layer):
    def __init__(self, units,alpha=0.01, activation=None, use_bias=True, kernel_initializer='glorot_uniform',custom_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, name=None, **kwargs):
        super(GLULayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.custom_initializer = tf.keras.initializers.get(custom_initializer)
        self.alpha = alpha
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.dense = tf.keras.layers.Dense(self.units, activation='elu', use_bias=self.use_bias,
                                             kernel_initializer=self.custom_initializer, bias_initializer=self.bias_initializer,
                                             kernel_regularizer=tf.keras.regularizers.L2(self.alpha), bias_regularizer=self.bias_regularizer,
                                             activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
                                             bias_constraint=self.bias_constraint, name=self.name + "_dense_a")

        self.dense_a = tf.keras.layers.Dense(self.units, activation='linear', use_bias=self.use_bias,
                                             kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                             kernel_regularizer=tf.keras.regularizers.L2(self.alpha), bias_regularizer=self.bias_regularizer,
                                             activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
                                             bias_constraint=self.bias_constraint, name=self.name + "_dense_a")
        self.dense_b = tf.keras.layers.Dense(self.units, activation=tf.sigmoid, use_bias=self.use_bias,
                                             kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                             kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                                             activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
                                             bias_constraint=self.bias_constraint, name=self.name + "_dense_b")
        super(GLULayer, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        a = self.dense_a(x)
        b = self.dense_b(x)
        weighted_x =  a * b + (1 - b) * x

        return weighted_x , b

    def compute_output_shape(self, input_shape):
        return  (input_shape[0], self.units) , (input_shape[0], self.units)

class LSTMSequenceRegressionWithGLU(tf.keras.layers.Layer):
    def __init__(self, units, alpha=0.01, **kwargs):
        super(LSTMSequenceRegressionWithGLU, self).__init__(**kwargs)
        self.units = units
        self.alpha = alpha
        self.lstm_cell = tf.keras.layers.LSTMCell(units)

    def build(self, batch_input_shape):
        self.w = self.add_weight(
            shape=(batch_input_shape[-1], self.units),
            initializer='random_normal',
            regularizer=tf.keras.regularizers.l2(self.alpha),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(LSTMSequenceRegressionWithGLU, self).build(batch_input_shape)

    def call(self, inputs, training=None):
        inputs = tf.matmul(inputs, self.w) + self.b
        inputs = tf.expand_dims(inputs, axis=1)
        outputs, _ = tf.keras.layers.RNN(self.lstm_cell, return_state=True)(inputs)
        outputs = tf.squeeze(outputs, axis=1)
        return outputs * tf.nn.sigmoid(outputs)


class RangeSubtractSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self,axis=-1):
        self.axis=axis
        super(RangeSubtractSoftmaxLayer, self).__init__()

    def call(self, inputs):
        # Calculate the range of each feature
        input_range = tf.reduce_max(inputs, axis=self.axis, keepdims=True) - tf.reduce_min(inputs, axis=-1, keepdims=True)

        # Subtract the range from each feature
        inputs_subtracted = inputs - input_range

        # Apply softmax to the subtracted inputs
        # softmax = tf.keras.activations.softmax(inputs_subtracted, axis=-1)
        # Apply exponential function to the subtracted x
        x_exp = tf.exp(inputs_subtracted)

        # Sum the exponential values along the specified axis
        x_sum = tf.reduce_sum(x_exp, axis=self.axis, keepdims=True)

        # Calculate the softmax output by dividing each exponential value by the sum
        softmax_output = x_exp / x_sum
        return softmax_output

class TAU(tf.keras.layers.Layer):
    def __init__(self,  units,dim,score_size,input_size,horizon,alpha,use_GCL=True,**kwargs):
        super(TAU, self).__init__(**kwargs)
        self.units =  units
        self.dim =  dim
        self.score_size =  score_size
        self.input_size =  input_size
        self.horizon =  horizon
        self.alpha = alpha
        self.use_GCL =use_GCL

    def build(self,input_shape):
        # self.X = [[GLULayer(self.units, alpha=self.alpha) for i in range(self.input_size)] for h in range(self.horizon)] # DynamicAttentionLayer(units=units,score_size=score_size,trainable = True)
        # if self.use_GCL :
        self.shared_ta_dense =layers.LSTM(units=self.units,return_sequences=True,recurrent_dropout=0.01 ,return_state=False) # layers.Dense(self.codes)
        # else:
        # self.shared_ta_dense =layers.Dense(units=self.units) # layers.Dense(self.codes)

        self.shared_ta_alpha = [layers.Dense(self.score_size) for h in range(self.horizon)  ]


    def call(self, inputs):
        weights_ta =[]
        contexts_ta = []
        for h in range(self.horizon):
            x = inputs[:,h,:,:]
            # if h==0:
            #     l_h = layers.Lambda(lambda xx: tf.where(xx == xx, tf.zeros_like(xx), tf.zeros_like(xx)))(va[:,0,:] )
            #     l_c = layers.Lambda(lambda xx: tf.where(xx == xx, tf.zeros_like(xx), tf.zeros_like(xx)))(va[:,0,:])
            # ta, l_h, l_c =  self.shared_ta_dense([x,l_h,l_c])
            ta =  self.shared_ta_dense(x)

            # if self.static_attention:
            #   ta =  self.shared_ta_alpha(ta)
            # else:
            ta =  self.shared_ta_alpha[h](ta)

            contexts_ta.append(tf.expand_dims(ta,1))

            # score_ta =  tf.nn.softmax(ta, axis=(-1))
            score_ta =  tf.nn.relu(ta )


            weights_ta.append(tf.expand_dims(score_ta,1))
            print('weights_ta',weights_ta)
        print('done')
        scores = layers.Concatenate(1)(weights_ta)
        contexts = layers.Concatenate(1)(contexts_ta)

        return contexts, scores


class MaxNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(MaxNormalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return inputs / (tf.reduce_max(inputs, axis=self.axis, keepdims=True)+.0000000001)

    def get_config(self):
        config = super(MaxNormalization, self).get_config()
        config.update({'axis': self.axis})
        return config

# import tensorflow_addons as tfa

import tensorflow as tf

class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout_rate=0.0):
        super(TemporalAttentionLayer, self).__init__()
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.attention_weights = None

        self.W1 = tf.keras.layers.Dense(units=num_units, activation='relu')
        self.W2 = tf.keras.layers.Dense(units=num_units, activation='relu')
        self.V = tf.keras.layers.Dense(units=1, activation=None)

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        """
        :param inputs: A tensor of shape (batch_size, sequence_length, num_features)
        :return: A tensor of shape (batch_size, num_features)
        """
        hidden = self.W1(inputs)
        hidden = self.dropout(hidden)
        hidden = self.W2(hidden)
        hidden = self.dropout(hidden)
        score = self.V(hidden)
        score = tf.nn.softmax(score, axis=1)

        self.attention_weights = tf.squeeze(score, axis=-1)

        weighted_inputs = tf.multiply(score, inputs)
        weighted_sum = tf.reduce_sum(weighted_inputs, axis=1)
        print('weighted_sum',weighted_sum)
        return weighted_sum

    def get_config(self):
        config = super(TemporalAttentionLayer, self).get_config()
        config.update({'num_units': self.num_units, 'dropout_rate': self.dropout_rate})
        return config


class Attention(  layers.Layer):
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

        input_dim = int(input_shape[0][-1])
        with K.name_scope(self.name ):
            # W in W*h_S.
            if self.score == self.SCORE_LUONG:
                self.luong_w = Dense(input_dim, use_bias=False, name='luong_w')
                # dot : last hidden state H_t and every hidden state H_s.
                self.luong_dot = layers.Dot(axes=[1, 2], name='attention_score')
            else:
                # Dense implements the operation: output = activation(dot(input, kernel) + bias)
                self.bahdanau_v = Dense(1, use_bias=False, name='bahdanau_v')
                self.bahdanau_w1 = Dense(input_dim, use_bias=False, name='bahdanau_w1')
                self.bahdanau_w2 = Dense(input_dim, use_bias=False, name='bahdanau_w2')
                self.bahdanau_repeat = layers.RepeatVector(input_shape[0][1])
                self.bahdanau_tanh = Activation('tanh', name='bahdanau_tanh')
                self.bahdanau_add = Add()

            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')

            # exp / sum(exp) -> softmax.
            self.softmax_normalizer = Activation('softmax', name='attention_weight')

            # dot : score * every hidden state H_s.
            # dot product. SUM(v1*v2). H_s = every source hidden state.
            self.dot_context = layers.Dot(axes=[1, 1], name='context_vector')

            # [Ct; ht]
            self.concat_c_h = layers.Concatenate(name='attention_output')

            # x -> tanh(w_c(x))
            self.w_c = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        # if not debug_flag:
        #     # debug: the call to build() is done in call().
        #     super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.units

    # def __call__(self, inputs, training=None, **kwargs):
    #     if debug_flag:
    #         return self.call(inputs, training, **kwargs)
    #     else:
    #         return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras. Supports:
            - Luong's multiplicative style.
            - Bahdanau's additive style.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: philipperemy, felixhao28.
        """
        # print('inputs',inputs[0])
        h_s = inputs[0]

        # if debug_flag:
        #     self.build(h_s.shape)
        h_t = self.h_t(h_s)

        if len(inputs)>1:
            d_c = inputs[1]
            h_t = layers.Add()([d_c,h_t] )
            # h_t = self.hNorm[t][i](dynamic_context)
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
        # for i in range (horizon):
        # alpha_s = tf.nn.relu (score)

        # alpha_s = tfa.activations.sparsemax(score)

        context_vector = self.dot_context([h_s, alpha_s])
        a_t = self.w_c(self.concat_c_h([context_vector, h_t]))

        if self.return_score:
            return a_t , alpha_s
        return a_t

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'units': self.units, 'score': self.score})
        return config




class MultiplyAttentionMasks(layers.Layer):
    def call(self, inputs):
        log_scores = tf.math.log(inputs)
        log_result = log_scores[0] * log_scores[1]
        result = tf.math.exp(tf.nn.softplus(log_result))
        return result

class ScaleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleAttention, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        attention_weights = tf.exp(inputs)
        attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=-1, keepdims=True)
        return attention_weights

class ElementwiseMulDivLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        A, B = inputs

        # Element-wise multiplication
        mul = A * B

        # Element-wise mean of A and B
        # mean = (A + B) / 2.0


        # Element-wise range of A and B
        range = tf.math.abs (A - B)

        # Element-wise division by mean
        div = mul / (range +0.000000001)

        return div


class CombineValuesLayer(tf.keras.layers.Layer):
    def __init__(self, threshold=0.1):
        super(CombineValuesLayer, self).__init__()
        self.threshold = threshold

    def call(self, inputs):
        A, B = inputs
        sorted_values1 = tf.sort(A, axis=-1, direction='DESCENDING')
        sorted_values2 = tf.sort(B, axis=-1, direction='DESCENDING')
        largestA = sorted_values1[:,:, self.threshold]
        largestB = sorted_values2[:, :,self.threshold]
        maximum_value = tf.maximum(largestA, largestB)

        max =  largestA
        # Perform the operation
        mask = tf.logical_or(A <= maximum_value, B <=  maximum_value)
        C = tf.where(mask, A * B, A + B)

        return C



class CheckKMaxmim(tf.keras.layers.Layer):
    def __init__(self, k=3):
        super(CheckKMaxmim, self).__init__()
        self.k = k
        self.threshold_k=0.1
        # self.threshold_variance=0.2
    def call(self, inputs):
        # print('inputs:::',inputs)

        sorted_values = tf.sort(inputs, axis=-1, direction='DESCENDING')
        threshold = sorted_values[:,  self.k]
        # Reshape the threshold tensor to have the same shape as inputs
        threshold = tf.expand_dims(threshold, axis=-1)
        threshold = tf.broadcast_to(threshold, tf.shape(inputs))
        max_mask = tf.math.greater_equal(inputs ,threshold )
        # mask =max_mask# tf.math.logical_and(max_mask, variance_mask)

        # threshold_k_tensor = tf.fill(tf.shape(threshold), self.threshold_k)
        #does the kth elemnt is = or > threshold_k
        # isLarger_mask = tf.math.greater_equal(inputs,threshold_k_tensor)
        # mask =  tf.math.logical_and(max_mask, isLarger_mask)

        # self.threshold_variance= tf.Variable(1., trainable=True,name='factor',dtype=tf.float32)

        # self.threshold_variance= tf.Variable(1., trainable=True,name='factor',dtype=tf.float32)

        # variance = tf.math.reduce_variance(inputs, axis=-1)
        # variance = tf.expand_dims(variance, axis=-1)
        # variance =  tf.broadcast_to(variance, tf.shape(inputs))
        # threshold_variance_tensor = tf.fill(tf.shape(variance), self.threshold_variance)
        # variance_mask = tf.math.greater_equal(variance, threshold_variance_tensor)



        # print('variancel',variance)
        # print('threshold_variance_tensor',threshold_variance_tensor)
        # print('max_mask',max_mask)
        # print('variance_mask',variance_mask)
        # mask =max_mask# tf.math.logical_and(max_mask, variance_mask)
        # Perform the operation
        # mask =  inputs <  threshold
        # mask = tf.math.greater(threshold, inputs)
        # print('mask:',mask)

        C = tf.where(max_mask,  inputs * 10 ,inputs )
        # C = tf.where(mask,  inputs * 10 ,inputs*.1 )
        # C = tf.where(mask,  inputs   ,inputs * 10)
        # print('C:',C)

        return C



class AmplificationLayer(layers.Layer):
    def __init__(self, exponent, **kwargs):
        super(AmplificationLayer, self).__init__(**kwargs)
        self.exponent = exponent

    def call(self, inputs):
        amplified_output = tf.pow(inputs, self.exponent)
        return amplified_output


class Attention(  layers.Layer):
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
                # dot : last hidden state H_t and every hidden state H_s.
                self.luong_dot = tf.keras.layers.Dot(axes=[1, 2], name='attention_score')
            else:
                # Dense implements the operation: output = activation(dot(input, kernel) + bias)
                self.bahdanau_v = tf.keras.layers.Dense(1, use_bias=False, name='bahdanau_v')
                self.bahdanau_w1 = tf.keras.layers.Dense(input_dim, use_bias=False, name='bahdanau_w1')
                self.bahdanau_w2 = tf.keras.layers.Dense(input_dim, use_bias=False, name='bahdanau_w2')
                self.bahdanau_repeat = tf.keras.layers.RepeatVector(input_shape[0][1])
                self.bahdanau_tanh = tf.keras.layers.Activation('tanh', name='bahdanau_tanh')
                self.bahdanau_add = tf.keras.layers.Add()

            self.h_t = tf.keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')

            # exp / sum(exp) -> softmax.
            self.softmax_normalizer = tf.keras.layers.Activation('softmax', name='attention_weight')

            # dot : score * every hidden state H_s.
            # dot product. SUM(v1*v2). H_s = every source hidden state.
            self.dot_context = tf.keras.layers.Dot(axes=[1, 1], name='context_vector')

            # [Ct; ht]
            self.concat_c_h = tf.keras.layers.Concatenate(name='attention_output')

            # x -> tanh(w_c(x))
            self.w_c = tf.keras.layers.Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        # if not debug_flag:
        #     # debug: the call to build() is done in call().
        #     super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.units

    # def __call__(self, inputs, training=None, **kwargs):
    #     if debug_flag:
    #         return self.call(inputs, training, **kwargs)
    #     else:
    #         return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        print('\n\n -------------------------- attention starts --------\n\n')
        """
        Many-to-one attention mechanism for Keras. Supports:
            - Luong's multiplicative style.
            - Bahdanau's additive style.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: philipperemy, felixhao28.
        """
        # print('inputs',inputs[0])
        print('>>> inputs',inputs)

        h_s = inputs[0]
        # print("h_s shape",h_s.shape)
        # if debug_flag:
        #     self.build(h_s.shape)
        h_t = self.h_t(h_s)
        # print("h_t shape",h_t.shape)
        print('>>>  h_t',h_t)

        if len(inputs)>1:

            d_c = inputs[1] #this is for the TAU
            print('d_c',d_c.shape)
            # i added this line to fix the shape mismatching issue 
            d_c = self.intermidiate_projector ( tf.expand_dims(d_c, axis=-1))
            d_c_broadcasted = tf.expand_dims(d_c, axis=1)
            d_c = tf.tile(d_c_broadcasted, [1, tf.shape(h_s)[1], 1])

            print('d_c',d_c.shape)
            print('h_s',h_s.shape)
            h_s = tf.keras.layers.Add()([d_c,h_s] )#h_t
            print('h_s',h_s.shape)
            
            # h_s = self.hNorm (h_s)
        if self.score == self.SCORE_LUONG:
            print('self.luong_w(h_s),',self.luong_w(h_s))
            # Luong's multiplicative style.
            score = self.luong_dot([h_t, self.luong_w(h_s)])


            # score = self.luong_dot([h_t, h_s])
        else:
            # Bahdanau's additive style.
            self.bahdanau_w1(h_s)
            a1 = self.bahdanau_w1(h_t)
            a2 = self.bahdanau_w2(h_s)
            a1 = self.bahdanau_repeat(a1)
            score = self.bahdanau_tanh(self.bahdanau_add([a1, a2]))
            score = self.bahdanau_v(score)
            score = K.squeeze(score, axis=-1)
        print('score',score)

        alpha_s = self.softmax_normalizer(score)

        context_vector = self.dot_context([h_s, alpha_s])
        print('context_vector',context_vector)
        print('self.concat_c_h([context_vector, h_t])',self.concat_c_h([context_vector, h_t]))
        a_t = self.w_c(self.concat_c_h([context_vector, h_t]))
        print('a_t',a_t)
        if len(inputs)>1:
            print('TA context > a_t',a_t)

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
        # self.hX = DynamicEmbeddings(units=self.horizon)

        # self.X1 = [layers.LSTM(units=self.input_size,return_sequences=True,recurrent_dropout=0.1 ,return_state=False,kernel_regularizer=tf.keras.regularizers.l2(self.alpha))  for i in range(self.input_size)]
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


