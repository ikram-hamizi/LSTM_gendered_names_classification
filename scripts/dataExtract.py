import pandas as pd

def get_data(lang="eng"):


	train = pd.read_csv("train_eng.csv") if lang == "eng" else pd.read_csv("train_rus.csv")
	test = pd.read_csv("test_eng.csv") if lang == "eng" else pd.read_csv("test_rus.csv")

	
	return train, test
