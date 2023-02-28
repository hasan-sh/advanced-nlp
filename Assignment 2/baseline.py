
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import FeatureHasher

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from utils_gabhoo import *
simplefilter("ignore", category=ConvergenceWarning)


def main(argv, arc):

    
    if arc!=2:
        raise("please provide path for data folder")

    
    else:
        data_path=argv[1]

  

              
    """train_ner = data_path+'/train_ner.tsv'
    test_ner = data_path+'/test_ner.tsv'
    train_ner_df = pd.read_csv(train_ner, delimiter='\t')
    test_ner_df = pd.read_csv(test_ner, delimiter='\t')"""

 
    train_file = data_path+'/en_ewt-up-train.conllu'
    test_file = data_path+'/en_ewt-up-test.conllu'
    train = read_data(train_file,save_to_csv=True)
    test = read_data(test_file, save_to_csv=True)
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    

    train_df=train[['token','target']] #BECAUSE ITS BASELINE
    test_df=test[['token','target']]


    train_df=downsample(train_df,820000)
    train_df.reset_index(inplace=True) #necessary after the down sample



    X_train,X_test = vectorize_hash_token(train_df.token,test_df.token)

    #here you do all the shit with the other features



    #this is the last thing to do as you want your helper to have all teh features and make better helping predicitons
    helper_train,helper_test=get_helper1(X_train,X_test,train_df['target'], test_df['target'])
    X_train['helper1']=helper_train
    X_test['helper1']=helper_test



    y_train,y_test=make_cat_label(train_df['target'], test_df['target'])
    #instantiate the model
    log_regression = LogisticRegression(penalty='l2')

    #fit the model using the training data
    log_regression.fit(X_train,y_train)

    #use model to make predictions on test data
    y_pred = log_regression.predict(X_test)

    f1 = f1_score(y_test,y_pred, average='weighted')
    prec = precision_score(y_test,y_pred, average='weighted')
    rec = recall_score(y_test,y_pred,average='weighted')
    print(f"{f1=}")
    print(f"{prec=}")
    print(f'{rec=}')



if __name__ == '__main__':

    main(sys.argv, len(sys.argv))