from keras.preprocessing.sequence import pad_sequences
import numpy as np

#function: converts data (Names) to vector representation (integers)
def name_to_vector(name, vocab, name_maxlen):
  vector = []
  for char in name:
    if char in vocab:
      integer_representation = vocab[char]
      vector.append(integer_representation)
      if len(vector) >= name_maxlen:
        break
  vector = pad_sequences([vector], padding='post', maxlen=name_maxlen)
  return vector

# function: convert the list-of-names data to NumPy arrays of integer indices to be fed to the models. The arrays are post-padded.
def data_to_matrix(names, matrix_size, vocab, name_maxlen):
  matrix = np.zeros((matrix_size, name_maxlen))  
  for i, name in enumerate(names):
    matrix[i] = name_to_vector(name, vocab, name_maxlen)
    
  return matrix

if __name__ == '__main__':
	pass
