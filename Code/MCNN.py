#!/usr/bin/env python
# coding: utf-8

# # Dependency

# In[1]:


import h5py
import os
import pickle

from tqdm import tqdm
from time import gmtime, strftime

import numpy as np
import math

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras import layers,Model

##

from sklearn.model_selection import KFold

import gc

import time
from sklearn.model_selection import KFold

#import import_test_ETC as load_data
# import import_test_nadp as load_data
import import_test_nadp_32_8 as load_data
#import import_test_nine as load_data


# # PARAM

# In[2]:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_type", type=str,default="ProtTrans", help='"BinaryMatrix" "MMseqs2" "ProtTrans"')
parser.add_argument("-n_dep","--num_dependent", type=int, default=5, help="the number of dependent variables")
parser.add_argument("-n_fil","--num_filter", type=int, default=1024, help="the number of filters in the convolutional layer")
parser.add_argument("-n_hid","--num_hidden", type=int, default=1000, help="the number of hidden units in the dense layer")
parser.add_argument("-bs","--batch_size", type=int, default=1024, help="the batch size")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,6,8,10], help="the window sizes for convolutional filters")
parser.add_argument("-n_feat","--num_feature", type=int, default=1024, help="the number of features")
parser.add_argument("-e","--epochs", type=int, default=20, help="the number of epochs for training")
parser.add_argument("-vm","--validation_mode", type=str, default="cross")

args=parser.parse_args()

DATA_LABEL=load_data.data_label()


NUM_DEPENDENT =args.num_dependent
MAXSEQ = NUM_DEPENDENT*2+1

DATA_TYPE = args.data_type
#"/BinaryMatrix" "/MMseqs2" "/ProtTrans"
NUM_FEATURE = args.num_feature

NUM_FILTER = args.num_filter
NUM_HIDDEN = args.num_hidden
BATCH_SIZE  = args.batch_size
WINDOW_SIZES = args.window_sizes




NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']


EPOCHS      = args.epochs

K_Fold = 5
VALIDATION_MODE=args.validation_mode
#"new388+41" "old227+17"
IMBALANCE="None"
#None
#SMOTE
#ADASYN
#RANDOM



#IMBLANCE="RANDOM"#"ADASYN" "RANDOM" "None" "SMOTE"
#SHUFFLE="Non-SHUFFLE"#Non-SHUFFLE


# In[3]:


import datetime

write_data=[]
a=datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(DATA_LABEL)
#write_data.append(DATASET)
write_data.append(DATA_TYPE)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
write_data.append(NUM_DEPENDENT)
write_data.append(IMBALANCE)

#write_data.append(IMBLANCE)
#write_data.append(SHUFFLE)


# # Time_log

# In[4]:


def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# In[5]:


import time
import math
def SAVEROC(fpr,tpr,AUC):
    data_to_save = {
        "fpr": fpr,
        "tpr": tpr,
        "AUC": AUC
    }
    
    with open("./PKL/MCNN_"+str(math.floor(time.time()))+"_"+DATA_TYPE+"_nfil_"+str(NUM_FILTER)+"_nhid_"+str(NUM_HIDDEN)+"_ws_"+str(WINDOW_SIZES)+VALIDATION_MODE+".pkl", "wb") as file:
        pickle.dump(data_to_save, file)


# # MCNN

# In[6]:


class DeepScan(Model):

	def __init__(self,
	             input_shape=(1, MAXSEQ, NUM_FEATURE),
	             window_sizes=[1024],
	             num_filters=256,
	             num_hidden=1000):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input(input_shape)
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
			 layers.Conv2D(filters=num_filters,
			               kernel_size=(1, window_size),
			               activation=tf.nn.relu,
			               padding='valid',
			               bias_initializer=tf.constant_initializer(0.1),
			               kernel_initializer=tf.keras.initializers.GlorotUniform()))
			self.maxpool.append(
			 layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
			                     strides=(1, MAXSEQ),
			                     padding='valid'))
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=0.7)
		self.fc1 = layers.Dense(
		 num_hidden,
		 activation=tf.nn.relu,
		 bias_initializer=tf.constant_initializer(0.1),
		 kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x


# # Main

# In[7]:


# Example usage:
x_train,y_train,x_test,y_test = load_data.MCNN_data_load(DATA_TYPE,NUM_CLASSES,NUM_DEPENDENT)


# In[8]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[9]:


def IMBALANCE_funct(IMBALANCE,x_train,y_train):
    if(IMBALANCE)=="None":
        return x_train,y_train
    else:
        from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
    
        # 將 x_train 的形狀重新整形為二維
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        print(x_train_2d.shape)
        print(y_train.shape)
        #print(y_train.shape)
        # 創建 SMOTE 物件
        if IMBALANCE=="SMOTE":
            imbalance = SMOTE(random_state=42)
        elif IMBALANCE=="ADASYN":
            imbalance = ADASYN(random_state=42)
        elif IMBALANCE=="RANDOM":
            imbalance = RandomOverSampler(random_state=42)
        
    
        # 使用 fit_resample 進行過採樣
        x_train_resampled, y_train_resampled = imbalance.fit_resample(x_train_2d, y_train)
    
        # 將 x_train_resampled 的形狀恢復為四維
        x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 1, MAXSEQ, NUM_FEATURE)
    
        print(x_train_resampled.shape)
        print(y_train_resampled.shape)
    
        x_train=x_train_resampled
        y_train=y_train_resampled
        
        del x_train_resampled
        del y_train_resampled
        del x_train_2d
        gc.collect()
    
        import tensorflow as tf
        y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
        return x_train,y_train


# In[10]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[11]:


def model_test(model, x_test, y_test):

    print(x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)
    #tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    display.plot()
    

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')
    SAVEROC(fpr,tpr,AUC)
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC


# In[12]:


if(VALIDATION_MODE=="cross"):
    time_log("Start cross")
    
    kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
    results=[]
    i=1
    for train_index, test_index in kfold.split(x_train):
        print(i,"/",K_Fold,'\n')
        # 取得訓練和測試數據
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)
        X_train,Y_train=IMBALANCE_funct(IMBALANCE,X_train,Y_train)
        # 重新建模
        model = DeepScan(
        num_filters=NUM_FILTER,
            num_hidden=NUM_HIDDEN,
            window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.build(input_shape=X_train.shape)
        # 在測試數據上評估模型
        history=model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            verbose=1,
            shuffle=True
        )
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, X_test, Y_test)
        results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
        i+=1
        
        del X_train
        del X_test
        del Y_train
        del Y_test
        gc.collect()
        
    mean_results = np.mean(results, axis=0)
    print(f'TP={mean_results[0]:.4}, FP={mean_results[1]:.4}, TN={mean_results[2]:.4}, FN={mean_results[3]:.4}, Sens={mean_results[4]:.4}, Spec={mean_results[5]:.4}, Acc={mean_results[6]:.4}, MCC={mean_results[7]:.4}, AUC={mean_results[8]:.4}\n')
    write_data.append(mean_results[0])
    write_data.append(mean_results[1])
    write_data.append(mean_results[2])
    write_data.append(mean_results[3])
    write_data.append(mean_results[4])
    write_data.append(mean_results[5])
    write_data.append(mean_results[6])
    write_data.append(mean_results[7])
    write_data.append(mean_results[8])


# In[13]:


if(VALIDATION_MODE=="independent"):
	if IMBALANCE!="None":
		x_train,y_train=IMBALANCE_HANDLING(IMBALANCE,x_train,y_train)
    #
	time_log("Start Model Train")
	model = DeepScan(
		num_filters=NUM_FILTER,
		num_hidden=NUM_HIDDEN,
		window_sizes=WINDOW_SIZES)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.build(input_shape=x_train.shape)
	model.summary()

	model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		shuffle=True,
	)

    # Save the trained model
    model_save_path = f'models/model_{DATA_TYPE}_{str(NUM_FILTER)}_{str(NUM_HIDDEN)}_{str(WINDOW_SIZES)}.h5'
    os.makedirs(model_save_path, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

	time_log("End Model Train")
	time_log("Start Model Test")
	TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC = model_test(model, x_test, y_test)
	write_data.append(TP)
	write_data.append(FP)
	write_data.append(TN)
	write_data.append(FN)
	write_data.append(Sens)
	write_data.append(Spec)
	write_data.append(Acc)
	write_data.append(MCC)
	write_data.append(AUC)
    
	time_log("End Model Test")


# In[14]:





def save_csv(write_data,a):
    import csv
    b=datetime.datetime.now()
    write_data.append(b-a)
    open_csv=open("NADP_32_8.csv","a")
    write_csv=csv.writer(open_csv)
    write_csv.writerow(write_data)
save_csv(write_data,a)


# In[ ]:




