#!/usr/bin/env python
# coding: utf-8

# # > Import

# In[1]:


from tqdm import tqdm
import numpy as np
import pickle


# In[2]:


import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)


# # > Prepare data

# ## >> Load vocabulary

# In[4]:


file_all_entity_id = open("data/all_entity_id.txt").read().splitlines()
file_all_relation_id = open("data/all_relation_id.txt").read().splitlines()
file_all_type_id = open("data/entity_type_id.txt").read().splitlines()


# ## >> Load dataset 
# (function only)

# In[5]:


def _load_batched_dataset_from_cache(batch_num):
    dataset_path = pickle.load(
        open("cache/dataset_path_{}".format(batch_num), "rb"))
    dataset_label = pickle.load(
        open("cache/dataset_label_{}".format(batch_num), "rb"))
    return dataset_path, dataset_label

def _save_batched_dataset_to_cache(dataset_path, dataset_label, batch_num):
    pickle.dump(dataset_path,
                open("cache/dataset_path_{}".format(batch_num), "wb"))
    pickle.dump(dataset_label,
                open("cache/dataset_label_{}".format(batch_num), "wb"))
    
def _load_batched_dataset_from_txt(batch_num):
    dataset_path = []
    dataset_label = []

    p_file = open(
        "data/new_positive_path_{}.txt".format(batch_num)).read().splitlines()
    n_file = open(
        "data/new_negative_path_{}.txt".format(batch_num)).read().splitlines()

    for file in [p_file, n_file]:
        for path in tqdm(file):
            paths = path.split('#')

            label = paths[-1]
            paths = [x.split() for x in paths[:-1]]

            dataset_path.append(paths)
            dataset_label.append(label)

    dataset_path = np.array(dataset_path).astype('int')
    dataset_label = np.array(dataset_label).astype('int')

    return dataset_path, dataset_label


# ## >> Split dataset 
# (function only)

# In[6]:


from sklearn.model_selection import train_test_split

def _save_split_dataset(split_dataset, batch_num):
    pickle.dump(split_dataset, open("cache/split_dataset_{}".format(batch_num), "wb"))
    
def _load_split_dataset(batch_num):
    return pickle.load(open("cache/split_dataset_{}".format(batch_num), "rb"))


# In[7]:


def train_eval_test_split(X, y, eval_ratio=0.2, test_ratio=0.2):

    n_data = len(y)
    n_eval = int(0.2 * n_data)
    n_test = int(0.2 * n_data)
    n_train = n_data - n_eval - n_test 
    
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=n_eval, random_state=88)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=n_test, random_state=88)
    
    return X_train, X_eval, X_test, y_train, y_eval, y_test


# ## >> Dataset loader

# In[8]:


def load_batched_dataset(batch_num):

    try:
        X_train, X_eval, X_test, y_train, y_eval, y_test = _load_split_dataset(batch_num)
    except:
        
        try:
            dataset_path, dataset_label = _load_batched_dataset_from_cache(batch_num)
        except:
            dataset_path, dataset_label = _load_batched_dataset_from_txt(batch_num)
            _save_batched_dataset_to_cache(dataset_path, dataset_label, batch_num)
        
        X_train, X_eval, X_test, y_train, y_eval, y_test = train_eval_test_split(dataset_path, dataset_label)

        split_dataset = (X_train, X_eval, X_test, y_train, y_eval, y_test)
        _save_split_dataset(split_dataset, batch_num)
        
    finally:
        return X_train, X_eval, X_test, y_train, y_eval, y_test


# Try to preprocess and cache it

# In[9]:


BATCH_COUNT = 72

for i in range(0, BATCH_COUNT):
    load_batched_dataset(i)


# In[10]:


# Sample one dataset batch

X_train, X_eval, X_test, y_train, y_eval, y_test = load_batched_dataset(1)
print("X_train :", X_train.shape)
print("X_eval :", X_eval.shape)
print("X_test :", X_test.shape)


# # > Model

# ## >> Define architecture

# In[11]:


from keras.layers import Input, Lambda, Embedding, Concatenate, LSTM, Dense, BatchNormalization, Activation


# In[12]:


N_CHANNEL = 3 # entity, entity type, relation

input_layer = Input(shape=(4, N_CHANNEL));


# In[13]:


n_users = 150000
n_entity = len(file_all_entity_id) + n_users
n_type = len(file_all_type_id)
n_relation = len(file_all_relation_id)

entity_emb_dim = 24
type_emb_dim = 8
relation_emb_dim = 8

input_entity = Lambda(lambda x: x[:, :, 0])(input_layer)
entity_embedding = Embedding(output_dim=entity_emb_dim, input_dim=n_entity)(input_entity)

input_type = Lambda(lambda x: x[:, :, 1])(input_layer)
type_embedding = Embedding(output_dim=type_emb_dim, input_dim=n_type)(input_type)

input_relation = Lambda(lambda x: x[:, :, 2])(input_layer)
relation_embedding = Embedding(output_dim=relation_emb_dim, input_dim=n_relation)(input_relation)

final_embedding = Concatenate()([entity_embedding, type_embedding, relation_embedding]);


# In[14]:


N_LSTM_UNIT = 16

lstm_layer = LSTM(N_LSTM_UNIT, activation='relu', dropout=0.5)(final_embedding);


# In[15]:


predictions = Dense(1)(lstm_layer)
predictions = BatchNormalization()(predictions)
prediction = Activation("sigmoid")


# ## >> Building model

# In[16]:


from keras.optimizers import Adam
from keras.models import Model

optimizer = Adam(lr=0.003)
model = Model(inputs=input_layer, outputs=predictions)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Let's checkout the final model summary first :)

# In[17]:


model.summary()


# ## >> Callbacks

# In[18]:


from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        plt.figure(figsize=(16, 16))

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.acc, label="accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()

        plt.show()


plot_learning = PlotLearning()


# In[19]:


early_stopping = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=3, min_delta=0.005, restore_best_weights=True)


# In[20]:


import datetime
import os 

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_folder = "logs/" + str(timestamp)

os.makedirs(log_folder)
filepath = log_folder + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

model_checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)


# # > Train model

# ## >> Batch generator

# In[21]:


from math import ceil

n_batches = 0
n_eval_batches = 0
batch_size = 4096

for i in range(0, 72):
    X_train, X_eval, X_test, y_train, y_eval, y_test = load_batched_dataset(i)
    n_batches += ceil(len(y_train) / batch_size)
    n_eval_batches += ceil(len(y_eval) / batch_size)


# In[22]:


def generate_batches(batch_count, batch_size):

    counter = 0
    while True:

        X_train, X_eval, X_test, y_train, y_eval, y_test = load_batched_dataset(counter)        
        counter = (counter + 1) % batch_count

        for cbatch in range(0, X_train.shape[0], batch_size):
            yield (X_train[cbatch:(cbatch + batch_size), :, :],
                   y_train[cbatch:(cbatch + batch_size)])
            
def generate_eval_batches(batch_count, batch_size):

    counter = 0
    while True:

        X_train, X_eval, X_test, y_train, y_eval, y_test = load_batched_dataset(counter)        
        counter = (counter + 1) % batch_count

        for cbatch in range(0, X_eval.shape[0], batch_size):
            yield (X_eval[cbatch:(cbatch + batch_size), :, :],
                   y_eval[cbatch:(cbatch + batch_size)])

batch_generator = generate_batches(batch_count=BATCH_COUNT, batch_size=batch_size)    
batch_eval_generator = generate_eval_batches(batch_count=BATCH_COUNT, batch_size=batch_size)


# model.fit(X_train,
#           y_train,
#           validation_data=(X_eval, y_eval),
#           batch_size=4096,
#           epochs=15,
#           callbacks=[early_stopping, model_checkpoint],
#           verbose=1)  # starts training

# ## >> Train

# In[23]:


model.fit_generator(generator=batch_generator,
                    validation_data=batch_eval_generator,
                    steps_per_epoch=n_batches,
                    validation_steps=n_eval_batches,
                    epochs=15,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)
