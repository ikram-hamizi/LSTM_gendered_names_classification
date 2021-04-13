#local classes
import sys
sys.path.append("./scripts")
import scripts

#python libraries
import pandas as pd
import numpy as np
from keras.models import load_model

#Load data
train, test = scripts.dataExtract.get_data() #csv english

x_test = test['Name']
y_test = test['Gender'].values.tolist() 


#Preprocess data
#Encode labels with integer representations
from sklearn.preprocessing import LabelEncoder

# intermediate step: get label encodings from training set labels
# Sort names by length before creating mini-batches to help the model to learn shorter sequences first.
s = train.Name.str.len().sort_values().index #Int64Index
train = train.reindex(s)

y_train = train['Gender'].values.tolist()
y_train = np.array(label_encoder.fit_transform(y_train))

label_encoder = LabelEncoder()
y_test = np.array(label_encoder.transform(y_test))


# Get Matrix of vector representation of names
total_test = len(test) #83288
total_vocab = len(unique)+2 #52+2
vocab_size = len(vocab) #52
name_maxlen = 15

matrix_train_y = scripts.preprocess.data_to_matrix(x_test, total_test, vocab, name_maxlen)


model1 = load_model('baselneLSTM.h5')
loss, accuracy = model1.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Baseline NN - Accuracy: {accuracy*100}%')


model2 = load_model('classicFeedFWD.h5')
loss, accuracy = model2.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Classic NN - Accuracy: {accuracy*100}%')


model2 = load_model('customLSTM.h5')
loss, accuracy = model3.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Best Custom NN - Accuracy: {accuracy*100}%')



