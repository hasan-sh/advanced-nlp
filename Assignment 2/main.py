
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

simplefilter("ignore")

N=800000 #downsample coeffitient

def main(argv, arc):

    if arc!=2:
        raise("please provide path for data folder")
    
    else:
        data_path=argv[1]

    print('Reading Data and Engineering features ...')
    train_df, test_df = read_conll(data_path)
    
    print('Encoding features ...')
    X_train, y_train, X_test, y_test = encode_features(train_df, test_df)
    
    print('Training ...')
    helper_train, helper_test = get_helper1(X_train, X_test, train_df['target'], test_df['target'])
    evaluate_helper1(helper_test, make_binary_label(test_df['target'])) 

    X_train['helper1'] = helper_train
    X_test['helper1'] = helper_test

    y_pred = logistic_reg(X_train,X_test,y_train,y_test)
    print('Done')
    
    evaluate(y_test, y_pred)
    
    #missing proper per class evaluation here


if __name__ == '__main__':

    main(sys.argv, len(sys.argv))