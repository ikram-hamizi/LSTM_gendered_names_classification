## LSTM_gendered_names_classification
Advanced Machine Learning - Innopolis University (Spring2021)

The libraries can be installed using **pip install -r requirements.txt**
The project can be tested on **Git Actions** but running the workflow defined in ```setup.yml```.

The Structure of the repository 
```bash
├── data                     <- Data files directory
│   └── Data1                <- Dataset 1 directory
│
├── notebooks                <- Notebooks for analysis and testing
│   └── eda                  <- EDA Notebook directory
│  
│
├── scripts                  <- Standalone scripts
│   ├── dataExtract.py       <- Data Extraction script
│   └── preprocess.py        <- Data to vector representation
│
├── src                      <- Code for use in this project.
│   ├── train.py             <- train script
│   └── test.py              <- model test script
│
├── requirements.txt                            
└── README.md     
```


Homework 1 Link: https://hackmd.io/@gFZmdMTOQxGFHEFqqU8pMQ/Bk4V79k4u
The task is to use deep learning, particularly LSTM (long-short term memory) models, to classify names based on their generally corresponding binary genders. Two models are compared to a Baseline LSTM: a Feed-forward Neural Network and a custom LSTM after hyperparameter tuning.
