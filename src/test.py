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
from keras.models import load_model

#Load data
train, test = dataExtract.get_data() #csv english

x_test = test['Name']
y_test = test['Gender'].values.tolist() 


#Preprocess data
#Encode labels with integer representations
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Intermediate step: get label encodings from training set labels
# a. Sort names in training set by length (like train.py)
s = train.Name.str.len().sort_values().index #Int64Index
train = train.reindex(s)
y_train = train['Gender'].values.tolist()
# b. fit encoder to training set and transform testing set
y_train = np.array(label_encoder.fit_transform(y_train))
y_test = np.array(label_encoder.transform(y_test))


# Get Matrix of vector representation of names
total_test = len(test) #83288
total_vocab = len(unique)+2 #52+2
vocab_size = len(vocab) #52
name_maxlen = 15

matrix_train_y = preprocess.data_to_matrix(x_test, total_test, vocab, name_maxlen)


model1 = load_model('baselneLSTM.h5')
print(model1.summary())
loss1, accuracy1 = model1.evaluate(matrix_test_x, y_test, verbose=0)
print(f'1) Baseline LSTM NN - Test Accuracy: {accuracy1*100}%')


model2 = load_model('classicFeedFWD.h5')
print(model2.summary())
loss2, accuracy2 = model2.evaluate(matrix_test_x, y_test, verbose=0)
print(f'2) Classic Feed-Forward NN - Test Accuracy: {accuracy2*100}%')


model3 = load_model('customLSTM.h5')
print(model3.summary())
loss3, accuracy3 = model3.evaluate(matrix_test_x, y_test, verbose=0)
print(f'3) Best Custom LSTM NN - Test Accuracy: {accuracy3*100}%')



