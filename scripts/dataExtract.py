import pandas as pd

def get_data(path="../data/", lang="eng"):
	
	assert path[-1]=="/", f"Path Error '{path}': Your path should end in a '/': e.g. '../data/'"
	
	train_path = path = "train_eng.csv"
	test_path = path "test_eng.csv"
	
	train = pd.read_csv(train_path) if lang == "eng" else pd.read_csv("train_rus.csv")
	test = pd.read_csv(test_path) if lang == "eng" else pd.read_csv("test_rus.csv")

	return train, test

if __name__ == '__main__':
	pass
