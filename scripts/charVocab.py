import numpy as np

def get_vocab(train):
  # Creating a vocabulary index from train
  train_np = train.to_numpy()
  unique = list(set("".join(train_np[:,0])))
  unique.sort() #len(unique) = 52 (26 lower + 26 UPPER)
  vocab = dict(zip(unique, range(1,len(unique)+1))) #character:index
  
  return vocab
