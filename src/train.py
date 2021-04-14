#local classes
import sys
# sys.path.append("../scripts")
import context
import scripts
import scripts.dataExtract as dataExtract
import scripts.preprocess as preprocess

#python libraries
import pandas as pd
import numpy as np

#Load data
train, _ = dataExtract.get_data() #csv english

#Preprocess data
#Sort names by length before creating mini-batches to help the model to learn shorter sequences first.
s = train.Name.str.len().sort_values().index #Int64Index
train = train.reindex(s)

x_train = train['Name']
y_train = train['Gender'].values.tolist()

from sklearn.preprocessing import LabelEncoder
#Encode labels with integer representations
label_encoder = LabelEncoder()
y_train = np.array(label_encoder.fit_transform(y_train))


# A. Creating a vocabulary index
train_np = train.to_numpy()
unique = list(set("".join(train_np[:,0])))
unique.sort() #len(unique) = 52 (26 lower + 26 UPPER)
vocab = dict(zip(unique, range(1,len(unique)+1))) #character:index

# Get Matrix of vector representation of names:
# Convert the list-of-names data to NumPy arrays of integer indices to be fed to the models. The arrays are post-padded.
total_train = len(train) #83288
total_vocab = len(unique)+2 #52+2
vocab_size = len(vocab) #52
name_maxlen = 15

matrix_train_x = preprocess.data_to_matrix(x_train, total_train, vocab, name_maxlen)

# B. Load pretrained char embeddings (GloVe)
#Usage: https://keras.io/examples/nlp/pretrained_word_embeddings/
# !wget https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt
import wget
url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt"
wget.download(url)

# B.1. Make a dict mapping GloVe chars to their NumPy vector representation
path_to_glove_file = "glove.840B.300d-char.txt"

#https://keras.io/examples/nlp/pretrained_word_embeddings/

# 1. Make a dict mapping GloVe chars to their NumPy vector representation
path_to_glove_file = "glove.840B.300d-char.txt"

embedding_vectors = {}
with open(path_to_glove_file, 'r') as f:
    for line in f:
        line_array = line.strip().split(" ")
        vector = np.array(line_array[1:], dtype=float)
        char = line_array[0]
        embedding_vectors[char] = vector

print("Found %s char vectors.", len(embedding_vectors))

embedding_dim = list(embedding_vectors.values())[0].shape[0]
print("Embedding Vector size =", embedding_dim)

# B.2. Prepare an embedding matrix to be used in the model
hits = 0
misses = 0

embedding_matrix_train_x = np.zeros((total_vocab, embedding_dim))  
for char, value in vocab.items():
	embedding_vector = embedding_vectors.get(char)
	if embedding_vector is not None:
		embedding_matrix_train_x[value] = embedding_vector
		hits+=1
	else:
		misses+=1

		print("(Embeddings) Converted %d chars (%d misses)" % (hits, misses))
		
# B.3. Apply PCA to reduce n_components from 300 to 50
from sklearn.decomposition import PCA
pca = PCA(n_components=50).fit(embedding_matrix_train_x)
embedding_matrix_train_x_pca = np.array(pca.transform(embedding_matrix_train_x))


# B.4. Initialize pretrained embedding layer
from tensorflow.keras.layers import Embedding
from keras.initializers import Constant

embedding_layer = Embedding(
    input_dim=embedding_matrix_train_x_pca.shape[0],  #len(train)+2 = 83290
    output_dim=embedding_matrix_train_x_pca.shape[1], #embedding_dim = 300
    embeddings_initializer=Constant(embedding_matrix_train_x_pca),
    trainable=False,
)



#C. Building and Training the models (with the best hyperparameters)
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

earlystopping = EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
reduceLRonplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.316, patience=3)
callbacks = [reduceLRonplateau, earlystopping]


from keras.preprocessing.sequence import pad_sequences

#function: converts data (Names) to vector representation (integers)
def name_to_vector(name):
  vector = []
  for char in name:
    if char in vocab:
      integer_representation = vocab[char]
      vector.append(integer_representation)
      if len(vector) >= name_maxlen:
        break


  vector = pad_sequences([vector], padding='post', maxlen=name_maxlen)
  return vector


import time
''' to run TensorBoard"
1- Run command in terminal: tensorboard --logdir='logs/' 
2- Copy address into a browser
'''
# Load the TensorBoard notebook extension
# %reload_ext tensorboard
# %tensorboard --logdir log

# C.1 >>>>>>>>>>>>>>>>>>>>>> BASELINE LSTM
# named_tuple = time.localtime() # get struct_time
# time_string = time.strftime("%d%m@%H:%M", named_tuple)
# MODEL_NAME = "Baseline_LSTM_{}".format(time_string)
# tensorboard = TensorBoard(log_dir='log/{}'.format(MODEL_NAME))

model1 = Sequential()
model1.add(Embedding(input_dim=vocab_size+1, input_length=15, output_dim=5)) #Documentation suggests to leave 0 for padding and add + 1 https://keras.io/api/layers/core_layers/embedding/
model1.add(LSTM(5)) #number of internal units
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer=Adam(0.001), loss='BinaryCrossentropy', metrics=["accuracy"])

baseline_lstm_start = time.time()
history_baseline = model1.fit(matrix_train_x, y_train,
          epochs=100,  batch_size=32, validation_split=0.2,
          verbose=2, callbacks=callbacks)#+[tensorboard])
baseline_lstm_end = time.time()

baseline_lstm_time = baseline_lstm_end - baseline_lstm_start
print("Time to train (Baseline LSTM):", baseline_lstm_time)

# get the best epoch
val_acc_per_epoch = history_baseline.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

model1.save("baselneLSTM.h5")

# C.2 >>>>>>>>>>>>>>>>>>>>>> CLASSIC FEED-FORWARD NN (After tuning hyperparameters using kerastuner)
# named_tuple = time.localtime() # get struct_time
# time_string = time.strftime("%d%m@%H:%M", named_tuple)
# MODEL_NAME = "custom_LSTM_{}".format(time_string)
# tensorboard = TensorBoard(log_dir='log/{}'.format(MODEL_NAME))
model2 = Sequential()
model2.add(embedding_layer)
model2.add(Flatten())
model2.add(Dense(160, activation='relu'))
model2.add(Dense(192, activation='relu'))
model2.add(Dense(160, activation='relu'))
model2.add(Dense(192, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

#best lr from hyperparameter tuning was: 0.001
model2.compile(optimizer=Adam(0.001), loss='BinaryCrossentropy', metrics=["accuracy"])

classic_fwd_start = time.time()
history_classic = model2.fit(matrix_train_x, y_train,
          epochs=100,  batch_size=32, validation_split=0.2,
          verbose=2, callbacks=callbacks)#+[tensorboard])
classic_fwd_end = time.time()

# print the time
classic_fwd_time = classic_fwd_end - classic_fwd_start
print("Time to train (Classic fwd NN):", classic_fwd_time)

# get the best epoch
val_acc_per_epoch = history_classic.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

model2.save("classicFeedFWD.h5")


# C.3 >>>>>>>>>>>>>>>>>>>>>> CLASSIC FEED-FORWARD NN (After tuning hyperparameters using kerastuner)
# named_tuple = time.localtime() # get struct_time
# time_string = time.strftime("%d%m@%H:%M", named_tuple)
# MODEL_NAME = "custom_LSTM_{}".format(time_string)
# tensorboard = TensorBoard(log_dir='log/{}'.format(MODEL_NAME))

model3 = Sequential()
model3.add(embedding_layer)
model3.add(LSTM(30))
model3.add(Dense(96, activation='relu'))
model3.add(Dense(224, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

#best lr from hyperparameter tuning was: 1e-2
model3.compile(optimizer=Adam(1e-2), loss='BinaryCrossentropy', metrics=["accuracy"])

custom_lstm_start = time.time()
history_best_custom = model3.fit(matrix_train_x, y_train,
          epochs=100,  batch_size=32, validation_split=0.2,
          verbose=2, callbacks=callbacks+[tensorboard])
custom_lstm_end = time.time()

custom_lstm_time = custom_lstm_end - custom_lstm_start
print("Time to train (Custom LSTM):", custom_lstm_time)

# get the best epoch
val_acc_per_epoch = history_best_custom.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

model3.save("customLSTM.h5") 
