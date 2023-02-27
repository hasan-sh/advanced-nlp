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


N=750000 #DOWNSAMPLING COEFFITIENT

#READ ANS MELT DF
def read_data(file_path, save_to_csv=False):    

    """
    This function reads a CoNLL-U format file and converts it into a pandas DataFrame.
    Each row in the DataFrame corresponds to a token in the file, and columns
    correspond to different features of the token, such as the token itself, its lemma, 
    part-of-speech tag, and syntactic dependency information.
    
    Parameters:
    file_path (str): The path to the input CoNLL-U format file.
    save_to_csv (bool): A boolean flag indicating whether to save the resulting DataFrame 
                        to a CSV file. Default is False.
                        
    Returns:
    df (pandas.DataFrame): A pandas DataFrame containing the token-level information from
                           the input file.
    """
    
    # Open and read the input file
    with open(file_path, 'r', encoding='utf-8') as f:
        train_data = f.read()
    
    # Split the file into individual documents, each separated by a blank line
    data = []
    for doc_i, doc in enumerate(train_data.split('\n\n')):
        doc = doc.split('\n')
        sentences = ''
        for line in doc:
            # Skip lines starting with '#' (comment lines)
            if line and line[0] != '#':
                line = line.split('\t')
                line.insert(0, str(doc_i))
                sentences += '\t'.join(line) + '\n'
        data.append(sentences)
    
    # Create a pandas DataFrame from the token-level data
    train_df = pd.DataFrame([x.split('\t') for sent in data for x in sent.split('\n') if x])
    
    # Rename the columns of the DataFrame
    train_df = train_df.rename(columns={
        0:'sent_id', 
        1:'token_id', 
        2:'token', 
        3:'lemma', 
        4:'POS', 
        5:'uni_POS',
        6:'morph_type', 
        7:'distance_head', 
        8:'dep_label', 
        9:'dep_rel', 
        10:'space', 
        11:'probbank'
    })
    
    # Convert the DataFrame from wide to long format
    df = train_df.melt(
        id_vars=[i for i in train_df.columns[:12]], 
        var_name="notneeded", 
        value_name="target"
    )
    
    # Drop the 'notneeded' column and any rows that contain missing values
    #df["sent_id"]=df['sent_id'].str.cat((df['notneeded'].astype(int)-12).astype(str) , sep="_" )
    df["repetion_id"]=df["notneeded"]-12
    df.drop(['notneeded'], axis=1, inplace=True)
    df = df[df['target'].notna()]
    
    # Optionally save the resulting DataFrame to a CSV file
    if save_to_csv:
        df.to_csv('./processed_data.tsv', sep='\t', index=False)
    
    # Return the resulting DataFrame

    return df

#PREPROCESSING

def make_binary_label(target_col):
  df = pd.DataFrame([0 if target=="_" or target=="V" else 1 for target in target_col],columns=['label'])
   #df= df.drop('target', axis=1)

  return df

def downsample(tokens_train,N):
  return tokens_train.drop(tokens_train[tokens_train['target']=="_"].sample(n=N).index)

def numerical_features(df,cols):
  """This function perform a preprocessing steps that consists of:
    - cast numerical features to int"""

  num_features = df[cols]
  num_features = num_features.astype(int)

  return num_features

def categorical_features(train,test,cols,N_out_feature=12):
  cat_features_train=train[cols]
  cat_features_test=test[cols]
  # Apply the hashing trick to the categorical features
  hasher = FeatureHasher(n_features=N_out_feature, input_type='string')
  hashed_features_train = hasher.fit_transform(cat_features_train.values.astype(str))
  hashed_features_test = hasher.transform(cat_features_test.values.astype(str))

  hashed_features_train= pd.DataFrame(hashed_features_train.toarray(),columns=["cat_"+str(i) for i in range(0,N_out_feature)])
  hashed_features_test = pd.DataFrame(hashed_features_test.toarray(),columns=["cat_"+str(i) for i in range(0,N_out_feature)])

  return hashed_features_train,hashed_features_test

def vectorize_hash_token(train_tokens,test_tokens,N_outfeature=65):

  hasher = FeatureHasher(n_features=N_outfeature,input_type='string')
  X_train = hasher.fit_transform(train_tokens)
  X_test = hasher.transform(test_tokens)
  # Convert the resulting sparse matrix to a dense matrix and concatenate with the numerical features
  X_train  = pd.DataFrame(X_train.toarray(),columns=["tok_"+str(i) for i in range(0,N_outfeature)])
  X_test  = pd.DataFrame(X_test.toarray(),columns=["tok_"+str(i) for i in range(0,N_outfeature)])

  return X_train,X_test

def make_cat_label(train_target,test_target):
  
  encoder = LabelEncoder() 
  y_all = np.concatenate([train_target, test_target], axis=0)
  # Fit the label encoder to the targets
  encoder.fit(y_all.astype(str))

  y_train = encoder.transform(train_target.astype(str))
  y_test = encoder.transform(test_target.astype(str))

  return y_train,y_test

def make_cat_label_OLD(train_target,test_target):
  
  encoder = LabelEncoder() 
  #y_all = np.concatenate([train_target, test_target], axis=0)
  # Fit the label encoder to the targets

  y_train = encoder.fit_transform(train_target.astype(str))
  y_test = encoder.transform(test_target.astype(str))

  return y_train,y_test


#MODEL 1
def get_helper1(X_train, X_test, y_train, y_test):
  """this function takes both train and test because the model that gives the information is the samerained on the training as well.
  It is adviced to use this at the end of preproessin for the first model to benefit from all the foruth future """

  #THIS DROPS THE TARGET CLASS AS WELL
  y_train= make_binary_label(y_train)
  y_test = make_binary_label(y_test)


  #print(X_train.shape,y_train.shape)
  
  
  log_regression = LogisticRegression(penalty='l2')
  #fit the model using the training data
  log_regression.fit(X_train,y_train)

  #use model to make predictions on test data
  helper1_train = log_regression.predict(X_train)

  helper1_test = log_regression.predict(X_test)

  return helper1_train,helper1_test


#EVALS
def logistic_reg(X_train,X_test,y_train,y_test):
  #instantiate the model
  log_regression = LogisticRegression(penalty='l2')

  #fit the model using the training data
  log_regression.fit(X_train,y_train)

  #use model to make predictions on test data
  y_pred = log_regression.predict(X_test)
  return y_pred

def evaluate_helper1(pred,y_test):
  f1 = f1_score(y_test,pred)
  arg_pred=pred.sum()
  nr_arg=y_test.sum()
  print(f"Score for the helper1 feature is {f1=}. ",arg_pred," out of the ", nr_arg, "args in test were predicted.")
  
def confusion_mtrx_binary(y_test,y_pred):
  cm = confusion_matrix(y_test, y_pred)
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation
  sns.color_palette("tab10")

  # labels, title and ticks
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
  ax.set_title('Confusion Matrix'); 
  ax.xaxis.set_ticklabels(['non_arg', 'arg']); ax.yaxis.set_ticklabels(['non_arg', 'arg']);
  return 0

def downsampling_study(train,test,start=100000,end=900000,step=50000):

  down_sample_size =  [x for x in range(start,end,step)]
  down_sample_size.reverse() #this is because theese represents how many rows it will remove, dus to have an increasing remaining size of train test we reverse it

  f1_scores=[]
  true_predicts=[]
  n_samples=[]
  for x in down_sample_size:
    print(f"{x=}")
    strain=train.copy()
    stest=test.copy()
    strain=downsample(strain,x)

    train_size=strain.shape[0]
    n_samples.append(train_size)
    print(f"{train_size=}")


    X_train=strain
    y_train=X_train.pop('label')
    X_test=stest
    y_test=X_test.pop('label')

    y_pred = logistic_reg(X_train,X_test,y_train,y_test)
    
    npred=y_pred.sum()
    true_predicts.append(npred)
    print(f"{npred=}")

    f1 = f1_score(y_test,y_pred, average='weighted')
    f1_scores.append(f1)
    print(f"{f1=}")
    print("-------------------X----------------")

  return n_samples,f1_scores,true_predicts

def plot_downsampling_study(n_sample,var,var_type):
  plt.plot(n_sample,var)
  plt.xlabel("Nr of instances")
  plt.ylabel(var_type)
  plt.title(var_type+" on nr of training intances")
  plt.xticks([x for x in range(100000,1000000, 200000)])
  plt.rcParams['figure.figsize'] = [3.8,3.8]
  #plt.xlim(0, 100e+4)

  # set the grid on
  plt.grid('on')

