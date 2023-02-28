from utils import *
from warnings import simplefilter
import sys
simplefilter("ignore")

N = 500000 # downsample coeffitient

import pandas as pd
def read_dfs():
    train_file = '../data/train_ner.tsv'
    test_file = '../data/test_ner.tsv'
    train_df = pd.read_csv(train_file, delimiter='\t')
    test_df = pd.read_csv(test_file, delimiter='\t')
    
    train_df = train_df[train_df['target'].notna()]
    test_df = test_df[test_df['target'].notna()]
    
    train_df = train_df[train_df["distance_head"]!="_"].reset_index(drop=True)
    test_df = test_df[test_df["distance_head"]!="_"].reset_index(drop=True)
    return train_df, test_df

def main(argv, arc):

    if arc!=2:
        raise("please provide path for data folder")
    
    else:
        data_path=argv[1]

    print('Reading Data and Engineering features ...')
    # train_df, test_df = read_conll(data_path)
    train_df, test_df = read_dfs()
    
    # print('Downsampling..')
    # train_df = downsample(train_df, N)
    
    print('Encoding features ...')
    X_train, y_train, X_test, y_test, y_test_inverted, mapping = encode_features(train_df, test_df)
    
    print('Training ...')
    helper_train, helper_test = get_helper1(X_train, X_test, train_df['target'], test_df['target'])
    evaluate_helper1(helper_test, make_binary_label(test_df['target'])) 

    X_train['helper1'] = helper_train
    X_test['helper1'] = helper_test

    y_pred = logistic_reg(X_train,X_test,y_train,y_test)
    print('Done')
    
    evaluate(y_test, y_pred)
    
    # Proper per class evaluation
    labels = list(set(y_train))

    report = classification_report(y_test, y_pred,
                                   target_names=[mapping[i] for i in labels])
    print(report)


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))