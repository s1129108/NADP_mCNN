from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

datalabel="NADP"

def data_label():
    return datalabel
# Fang/experiments/get_complete_dataset/NAD_195/Onehot
def MCNN_data_load(DATA_TYPE,NUM_CLASSES,NUMDEPENDENT):
    MAXSEQ=NUMDEPENDENT*2+1
    path_x_train = "./protTrans_dataset/data.npy"
    path_y_train = "./protTrans_dataset/label.npy"
    print(path_x_train)
    print(path_y_train)
    x,y=data_load(path_x_train,path_y_train,NUM_CLASSES)
    # path_x_test = "/mnt/D/jupyter/Fang/experiments/get_complete_dataset/NADP_40/"+DATA_TYPE+"/data.npy"
    # path_y_test = "/mnt/D/jupyter/Fang/experiments/get_complete_dataset/NADP_40/"+DATA_TYPE+"/label.npy"
    path_x_test = "./test/protTrans/data.npy"
    path_y_test = "./test/protTrans/label.npy"
    print(path_x_test)
    print(path_y_test)
    x_test,y_test=data_load(path_x_test,path_y_test,NUM_CLASSES)
    
    return(x,y,x_test,y_test)

def data_load(x_folder, y_folder,NUM_CLASSES,):
    x_train=np.load(x_folder)
    y_train=np.load(y_folder)
    

    y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    gc.collect()
    
    return x_train, y_train