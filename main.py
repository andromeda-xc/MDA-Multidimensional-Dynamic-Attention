
# fix random seed for reproducibility


# import tensorflow  
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

import numpy as np
import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(123)
tf.random.set_seed(1234) 

from keras import backend as k
config =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=config)
# k.set_session(sess)
# from models1 import *
# from models_multistep import *
from models_multistep_time_distrbuted import *
# from keras.utils.vis_utils import plot_model
# from keras import callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import csv
import pandas as pd
from time import time
# import tensorflow.compat.v1 as tf

# import tensorflow as tf
# from keras import backend as K
from tqdm import tqdm


# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)


# def reset_seed(seed):
#         # tf.disable_v2_behavior()
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#         os.environ['ERAS_BACKEND'] = 'tensorflow'
#         tf.keras.backend.clear_session()
#         # tf.random.set_seed(seed)
#         # tf.set_random_seed(seed)
#         np.random.seed(seed)
#         # config = tf.ConfigProto()
#         # config = tf.compat.v1.ConfigProto()
#         # config.gpu_options.allow_growth = True
#         # sess = tf.compat.v1.Session(config=config)
#         # # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

#         # # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#         # K.set_session(sess)


def start (data , params,deNormalize_min_max='') :
    x_train, y_train, x_val,y_val , x_test,y_test ,scaler_y= data

    # Print shapes
    print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)  # (30652, 10, 8), (30652, 2)
    print('x_val shape:', x_val.shape, 'y_val shape:', y_val.shape)     # (8758, 10, 8), (8758, 2)
    print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)    # (4379, 10, 8), (4379, 2)

    # reset_seed(params['seed']) 


    model_name = params['model_name']

    # Create directory to save results
    dir_ = '%s/%s/%s_Tx_%s_Ty_%s_inpvar_%s_run_%s'\
                %(params['dataset_name'], params['pred_type'], model_name, params['time_steps'], params['horizon'], params['no_varibles'], params['seed'])
    
    if not os.path.exists(dir_):
        os.makedirs(dir_) 
    # Save parameters
    with open('%s/pameters.csv' %(dir_), 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
        
      

    # MAPE
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Plot Ground Truth, Model Prediction
    def actual_pred_plot (y_actual, y_pred, n_samples = 60):
        
        # Shape of y_actual, y_pred: (8758, Ty)
        plt.figure()
        plt.plot(y_actual[ : n_samples, -1])  # 60 examples, last prediction time step
        plt.plot(y_pred[ : n_samples, -1])    # 60 examples, last prediction time step
        plt.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
        plt.savefig('%s/actual_pred_plot.png'%(dir_))
        print("Saved actual vs pred plot to disk")
        plt.close()

    # Correlation Scatter Plot
    def scatter_plot (y_actual, y_pred):
        
        # Shape of y_actual, y_pred: (8758, Ty)
        plt.figure()
        plt.scatter(y_actual[:, -1], y_pred[:, -1])
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=4)
        plt.title('Predicted Value Vs Actual Value')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.savefig('%s/scatter_plot.png'%(dir_))
        print("Saved scatter plot to disk")
        plt.close()
            

    # Evaluate Model
    def evaluate_model (x_data, y_data, dataset):
        
        print('---------------- evaluate model ----------------')


        # x_train: (30652, 10, 8), y_train: (30652, 2, 1)
        # s_data = x_data.transpose(0, 2, 1)   # (30652, 8, 10)
        
        # s0_data = np.zeros((y_data.shape[0], h_s))
        # c0_data = np.zeros((y_data.shape[0], h_s))
        # yhat0_data = np.zeros((y_data.shape[0], 1))
        if dataset== "test":
            start_time=time()        

        print('here====')
        # y_data_hat = model.predict([x_data, s_data, s0_data, c0_data, yhat0_data], batch_size = batch_size)
        y_data_hat = model.predict( x_data  )
        y_data_hat = np.array(y_data_hat)     # (2, 30652, 1)
        y_data = np.array(y_data)
        print('y_data_hat',y_data_hat.shape)
        if dataset== "test":
            print("Total testing time: ", time()-start_time)   
        

        
        if model_name== 'STAM':     # For Ty = 1, y_data_hat shape is already (30652, 1)
            y_data_hat = y_data_hat.swapaxes(0,1)     # (30652, 2, 1)
            y_data = y_data.swapaxes(0,1)
        y_data_hat = y_data_hat.reshape((y_data_hat.shape[0], params['horizon']))    # (30652, 2)

        if params['dataset_name']== 'qld' or  params['dataset_name']== 'NSW' :
            y_data_hat = deNormalize_min_max(y_data_hat)       

        else:
            y_data_hat = scaler_y.inverse_transform(y_data_hat)       
        np.save("%s/y_%s_predictions"%(dir_, dataset), y_data_hat)  # y_val_hat_betas

        y_data = y_data.reshape((y_data.shape[0], params['horizon']))      # (30652, 2)
        if params['dataset_name'] == 'qld' or  params['dataset_name']== 'NSW' :
            y_data = deNormalize_min_max(y_data)       

        else:
            y_data = scaler_y.inverse_transform(y_data)
        np.save("%s/y_%s_actuals"%(dir_, dataset), y_data)  # y_val_hat_betas
        
        
        # Selecting the output only for Ty timestep
        y_data_hat_Ty = y_data_hat [:, (params['horizon'] - 1)]   # (30652, )
        y_data_Ty = y_data [:, (params['horizon'] - 1)]    # (30652, )
        
        metric_dict = {}  # Dictionary to save the metrics
        
        data_rmse = sqrt(mean_squared_error(y_data_Ty, y_data_hat_Ty))
        metric_dict ['rmse'] = data_rmse 
        print('%s RMSE: %.4f' %(dataset, data_rmse))
        
        data_mae = mean_absolute_error(y_data_Ty, y_data_hat_Ty)
        metric_dict ['mae'] = data_mae
        print('%s MAE: %.4f' %(dataset, data_mae))
        
        data_r2score = r2_score(y_data_Ty, y_data_hat_Ty)
        metric_dict ['r2_score'] = data_r2score
        print('%s r2_score: %.4f' %(dataset, data_r2score))
            
        # Save metrics
        with open('%s/metrics_%s.csv' %(dir_, dataset), 'w', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in metric_dict.items():
                writer.writerow([key, value])
        
        # Save Actual Vs Predicted Plot and Scatter PLot for test set
        if dataset == 'test':
            actual_pred_plot (y_data, y_data_hat)
            scatter_plot (y_data, y_data_hat)
            
        return metric_dict

    #####--------------------------------------------------------------------------------------------------------------------------------############################

    def get_proper_inputs(x_data , y_data):
        s_data = x_data.transpose(0, 2, 1)
        s0_data = np.zeros((y_data.shape[0], params['h_s_lstm']))
        c0_data = np.zeros((y_data.shape[0], params['h_s_lstm']))
        yhat0_data = np.zeros((y_data.shape[0], 1))


        if model_name == 'TFT':
            x_data = [x_data,s0_data,c0_data]  #for TFT
        elif model_name == 'STAM':
            x_data = [x_data,s_data, s0_data,c0_data,yhat0_data]     
            y_data = list(y_data.swapaxes(0,1))    # Ty numpy lists each (30562, 1)

        # elif model_name =='MAFS_extend_parlallel' or model_name == 'MAFS_extend_sequential':
        #     x_data = [x_data, s0_data,c0_data]     
        elif model_name in ['NN' , 'CNN', 'CNNLSTM','MAFS_extend_parlallel','MAFS_extend_sequential']:
            x_data = [x_data ]     


        # print('xdata length ' , x_data.shape)
        print('xdata length ' , len(x_data))
        # print('y_data',y_data.shape)
        
        return x_data, y_data


    if model_name =='TFT':
        model,weights_model = TFT(params).build_model()
    elif model_name == 'STAM':
        model , weights_model = STAM(params).build_model()

    elif model_name == 'MAFS':
        model , weights_model = MAFS(params).build_model()
    elif model_name =='MAFS_extend_parlallel' or model_name =='MAFS_extend_sequential' :
        
        model , weights_model = MAFS_extend(params).build_model()
        no_parameters =  model.count_params()
        print('model model:',no_parameters)
        print('weights_model model:',weights_model.count_params())
        
    elif model_name =='NN'  :
        model  = NN(params).build_model()
    elif model_name =='CNN'  :
        model  = CNN(params).build_model()
    elif model_name =='CNNLSTM'  :
        model  = CNNLSTM(params).build_model()

        
    # model , weights_model= Parallel_Attention(params).build_model()

    # Attention Weights Model

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    callback_lists = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'],verbose=True, restore_best_weights=True),

        # callbacks.LearningRateScheduler(scheduler,verbose=True)
    ]



    x_train, y_train = get_proper_inputs(x_train, y_train)
    print('done')
    x_val, y_val = get_proper_inputs(x_val, y_val)
    print('now')
    
    # outputs_train = list(y_train.swapaxes(0,1))    # Ty numpy lists each (30562, 1)
    # outputs_val = list(y_val.swapaxes(0,1))        # Ty numpy lists each (8758, 1)
    outputs_train = y_train # list(y_train.swapaxes(0,1))    # Ty numpy lists each (30562, 1)
    outputs_val = y_val # list(y_val.swapaxes(0,1))        # Ty numpy lists each (8758, 1)

    no_parameters =  model.count_params()
    print('no_parameters:',no_parameters)

    with open(dir_+'/no_parameters.txt', 'w') as f:
        f.write(str(np.round(no_parameters,3)))

    start_time=time()

    hist = model.fit (x_train, outputs_train,
                    batch_size = params['batch_size'] ,
                    epochs = params['epochs'] ,
                    callbacks = callback_lists,   # Try Early Stopping
                    verbose = 2,
                    #   shuffle = True,
                    shuffle = False,
                    #   validation_split= .1)
                    validation_data=(x_val , outputs_val),
                    workers=10,use_multiprocessing=True)
    end_time = time()

    ###################



 

    if model_name not in ['NN' , 'CNN', 'CNNLSTM']:
        weights_model.set_weights(model.get_weights())
           
    training_time = np.round( end_time-start_time,3)
    with open(dir_+'/training_time.txt', 'w') as f:
        f.write(str(np.round(training_time,3)))
    print("Per epoch train time:",(end_time-start_time)/params['epochs'] )
    with open(dir_+'/time_per_epoch.txt', 'w') as f:
        f.write(str(np.round((end_time-start_time)/params['epochs'] ,3)))
    # Plot
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure()
    plt.plot(loss, label ='Training Set')
    plt.plot(val_loss,label= 'Validation Set')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(  loc='upper right')
    plt.savefig('%s/loss_plot.png'%(dir_))
    print("Saved loss plot to disk") 
    plt.close()

    # Save Data
    loss = pd.DataFrame(loss).to_csv('%s/loss.csv'%(dir_))    # Not in original scale 
    val_loss = pd.DataFrame(val_loss).to_csv('%s/val_loss.csv'%(dir_))  # Not in original scale


    #####fix these for all inputs of different data test..tran etc 

    # Evaluate Model - Train, Validation, Test Sets
    x_test, y_test = get_proper_inputs(x_test, y_test)

    train_metrics = evaluate_model (x_train, y_train, 'train')
    val_metrics = evaluate_model (x_val, y_val, 'val')
    test_metrics = evaluate_model (x_test, y_test, 'test')

    # Get Attention Weights
    def get_weights_STAM  (x_data, y_data, dataset):
        
        # Get numpy array of alphas in shape (30652, 10, 2) from list of 2(Ty) elements each element (30652, 10, 1)
        def alphas_array(prob_list):
            
            prob = np.array(prob_list)
            print('prob',prob.shape)
            prob = prob.swapaxes(0, 1)
            prob = prob.swapaxes(1, 2)
            prob = prob.reshape((prob.shape[0], prob.shape[1], prob.shape[2]))

            # if len (prob.shape) ==3:
            #     prob = prob.reshape((prob.shape[0], prob.shape[1], prob.shape[2]))
            # elif   len (prob.shape) ==4:

            return prob


        # s_data = x_data[1] #x_data.transpose(0, 2, 1)   # (30652, 8, 10)
        
        # s0_data = np.zeros((y_data.shape[0], h_s))
        # c0_data = np.zeros((y_data.shape[0], h_s))
        # yhat0_data = np.zeros((y_data.shape[0], 1))
        
        # y_data_hat_prob = weights_model.predict([x_data, s_data, s0_data, c0_data, yhat0_data], batch_size = batch_size)
        y_data_hat_prob = weights_model.predict(  x_data, batch_size = params['batch_size'] )
        len_list = len(y_data_hat_prob)   # List of 6 elements  
        
        # alphas list: elements 1, 4, each element (30652, 10, 1)
        y_data_hat_alphas = [y_data_hat_prob[i] for i in range(1, len_list,3)]
        y_data_hat_alphas = alphas_array (y_data_hat_alphas)      # (30652, 10, 2)
        np.save("%s/y_%s_hat_alphas"%(dir_, dataset), y_data_hat_alphas)  # y_val_hat_alphas
        
        # betas list: elements 2, 5, each element (30652, 8, 1)
        y_data_hat_betas = [y_data_hat_prob[i] for i in range(2, len_list, 3)]
        y_data_hat_betas = alphas_array (y_data_hat_betas)     # (30652, 8, 2)
        np.save("%s/y_%s_hat_betas"%(dir_, dataset), y_data_hat_betas)  # y_val_hat_betas
        
        return y_data_hat_alphas, y_data_hat_betas




    def tft_weights(x_data):
        # _,spatial_weights,temporal_attention = weights_model.predict(x_train) 
        _,spatial_weights ,temporal_attention = weights_model.predict(x_data) 
        print('spatial_weights',spatial_weights.shape)
        spatial_weights = np.reshape(spatial_weights , (spatial_weights.shape[0],x_data[0].shape[1]* x_data[0].shape[2])  )
        # spatial_weights = spatial_weights.mean(0)
        spatial_weights = pd.DataFrame(spatial_weights)
        spatial_weights.to_csv("%s/%s_score.csv"%(dir_, dataset) )
        print(spatial_weights )

        print('temporal_attention',temporal_attention.shape)
        temporal_attention = np.reshape(temporal_attention , (temporal_attention.shape[0],x_data[0].shape[1]* x_data[0].shape[1])  )
        temporal_attention = pd.DataFrame(temporal_attention)

        print('temporal_attention',temporal_attention)

        temporal_attention.to_csv("%s/%stemporal_attention.csv"%(dir_, dataset) )


    for dataset in ['valid','test']:#'train',
        print(dataset)

        if dataset == 'train':
            x_data = x_train
            y_data = y_train
        elif dataset == 'valid':
            x_data = x_val    
            y_data = y_val
        elif dataset =='test':
            x_data = x_test
            y_data = y_test
                
            
        if model_name =='TFT':
            tft_weights(x_data)

        elif model_name == 'STAM':
            # Attention Weights - Train, Validation, Test Sets
            alphas,  betas = get_weights_STAM(x_data, y_data, dataset) 
    

        elif model_name == 'MAFS' :
            _,score = weights_model.predict(x_data) 
            print('score',score.shape)
            print('x_data[0]',x_data[0].shape)

            np.save("%s/%s_score"%(dir_, dataset), score)  # y_val_hat_alphas

            score = np.reshape(score , (score.shape[0],x_data.shape[1]* x_data.shape[2])  )
            # spatial_weights = spatial_weights.mean(0)
            score = pd.DataFrame(score)
            # score.to_csv(dir_+'/att_weights.csv')
            # print(score )

        elif     model_name =='MAFS_extend_parlallel' or model_name =='MAFS_extend_sequential' :
            
            # _,score,gate_v, gate_t = weights_model.predict(x_data) 
            '''
            if params['dataset_name'] == 'qld' :

                y_hat = []
                score = []
                gate_V = []
                gate_T =[]
                

                # for id in range(x_data[0].shape[0]):
                for id in tqdm(range(x_data[0].shape[0]), desc = 'predicting for testexample '):    
                    example = [x_data[i][id:id+1] for i in range(len(x_data)) ] 
                    pred = weights_model.predict(example)
                    if id ==0:
                        y_hat, score, gate_V ,gate_T = pred[0],pred[1],pred[2],pred[3]
                    else:
                        y_hat = np.concatenate([y_hat,pred[0]],axis=0 )
                        score = np.concatenate([score,pred[1]],axis=0 )  
                        gate_V = np.concatenate([gate_V,pred[2]],axis=0 )  
                        gate_T = np.concatenate([gate_T,pred[3]],axis=0 )  
                    
                out = [y_hat, score, gate_V,gate_T]
            else:
                out = weights_model.predict(x_data,batch_size = params['batch_size']) 
            print(len(out))
            '''
            out = weights_model.predict(x_data,batch_size = params['batch_size']) 
            
            # 
            len_list = len(out)
            no_items_in_list = len_list//params['horizon']

            y_hat = out[0]
            score = out[1]
            print('y_hat',y_hat.shape)
            print('score',score.shape)
            np.save("%s/%s_score"%(dir_, dataset), score)  # y_val_hat_alphas

            # if params['use_TA']  and  params['use_VA']  :
            #     gate_V = out[2]
            #     gate_T = out[3]
            #     np.save("%s/%s_gate_V"%(dir_, dataset), gate_V)  # y_val_hat_alphas
            #     np.save("%s/%s_gate_T"%(dir_, dataset), gate_T)  # y_val_hat_alphas
            # elif params['use_TA']  and  not params['use_VA']  :
            #     gate_T = out[2]
            #     np.save("%s/%s_gate_T"%(dir_, dataset), gate_T)  # y_val_hat_alphas
            # elif params['use_VA']  and  not params['use_TA']  :
            #     gate_V = out[2]
            #     np.save("%s/%s_gate_V"%(dir_, dataset), gate_V)  # y_val_hat_alphas
            '''

            #yaht
            y_hat =  [out[i] for i in range(0, len_list,no_items_in_list)]
            y_hat = np.array(y_hat)
            print("y_hat:",y_hat.shape)
            y_hat = y_hat.swapaxes(0, 1)

            print("y_hat:",y_hat.shape)
            # y_data_hat_alphas = alphas_array (y_data_hat_alphas)      # (30652, 10, 2)
            # np.save("%s/y_%s_hat_alphas"%(dir_, dataset), y_data_hat_alphas)  # y_val_hat_alphas


            #score:  
            score = [out[i] for i in range(1, len_list,no_items_in_list)]
            score = np.array(score)
            print("score:",score.shape)
            score = score.swapaxes(0, 1)
            np.save("%s/%s_score"%(dir_, dataset), score)  # y_val_hat_alphas


            

            #gate_V:
            gate_V = [out[i] for i in range(2, len_list,no_items_in_list)]
            gate_V = np.array(gate_V)
            gate_V = gate_V.swapaxes(0, 1)
            np.save("%s/%s_gate_V"%(dir_, dataset), gate_V)  # y_val_hat_alphas
            print(gate_V.shape)              
            # gate_T: 
            gate_T = [out[i] for i in range(3, len_list,no_items_in_list)]
            gate_T = np.array(gate_T)
            gate_T = gate_T.swapaxes(0, 1)
            np.save("%s/%s_gate_T"%(dir_, dataset), gate_T)  # y_val_hat_alphas
            print(gate_T.shape)           
            '''

    return '-----------------'+ 'DONE'+'---------------'  