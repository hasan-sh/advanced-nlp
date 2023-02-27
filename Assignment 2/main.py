
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

    

    train_df=train[train["distance_head"]!="_"].reset_index(drop=True)#like this reset_index(drop=True)
    N=800000 #downsample coeffitient
    train_df=downsample(train_df,N)
    train_df.reset_index(inplace=True) #necessary after the removing rows

    test_df=test[test["distance_head"]!="_"].reset_index(drop=True)#like this reset_index(drop=True)


    print("CHECK DIM: ",train_df.shape,test_df.shape)
    #get vectorized tokens
    train_tokens,test_tokens = vectorize_hash_token(train_df.token,test_df.token)


    #get numerical features
    cols_num=['sent_id', 'token_id', 'distance_head']
    train_num=numerical_features(train_df,cols_num)
    test_num=numerical_features(test_df,cols_num)

    #categorical features
    cols_cat_features = ['POS', 'uni_POS', 'morph_type',
                        'dep_label', 'dep_rel', 'space', 'probbank']
    train_cat,test_cat=categorical_features(train_df, test_df, cols_cat_features)

    
    #concat everything into X_train
    X_train=pd.concat([train_tokens,train_cat,train_num],axis=1) #here you concat with all the shit
    X_test=pd.concat([test_tokens,test_cat,test_num],axis=1)
    helper_train,helper_test=get_helper1(X_train, X_test, train_df['target'], test_df['target'])

    evaluate_helper1(helper_test,make_binary_label(test_df['target'])) #this is a plus to check the down sampling

    X_train['helper1']=helper_train
    X_test['helper1']=helper_test


    #Turn targets into categorical labels
    y_train,y_test=make_cat_label(train_df['target'],test_df['target'])


    #train and predict
    y_pred=logistic_reg(X_train,X_test,y_train,y_test)

    #evaluation

    f1 = f1_score(y_test,y_pred, average='weighted')
    prec = precision_score(y_test,y_pred, average='weighted')
    rec = recall_score(y_test,y_pred,average='weighted')
    print(f"{f1=}")
    print(f"{prec=}")
    print(f'{rec=}')

    #missing proper per class evaluation here





if __name__ == '__main__':

    main(sys.argv, len(sys.argv))