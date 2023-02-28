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
import stanza

nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')

N=750000 #DOWNSAMPLING COEFFITIENT

def get_named_entities(sentence):
    """
    Get the named entity for each token in a data frame, and recombine the named entities into a list
    of named entities for each sentence.

    Args:
        df (pandas.DataFrame): A data frame with columns 'sent_id', 'token', and 'ner', where each row contains
        a token and its named entity.

    Returns:
        dict: A dictionary where the keys are the sent_ids and the values are lists of named entities, where each named entity corresponds
        to one token in the original sentence order.

    """
    # Use spaCy to extract named entities from the sentence
    doc = nlp(sentence)
    named_entities_for_sentence = [(ent.text, ent.type) for ent in doc.ents]
    
    # Return the resulting dictionary of named entities
    return named_entities_for_sentence

def assign_ne(row, named_enities):
    """
    Assigns named entity labels to a token based on a list of named entities.

    Args:
        row (pandas.Series): A pandas Series representing a row in a DataFrame with the following columns:
            - token: the token to be labeled
            - ... (additional columns not used by this function)
        named_entities (list): A list of named entities, where each named entity is a tuple with the following format:
            - enitiy text
            - enitiy type

    Returns:
        str: The label of the named entity that the token belongs to, or '_' if the token does not belong to
        any named entity in the list.
    """
    named_entity = list(filter(lambda x: row.token in x[0], named_enities))
    if named_entity:
        return named_entity[0][1]
    else:
        return '_'

def add_bigrams(df):
    """
    Adds columns for token and pos bigrams to a DataFrame containing token and pos columns.

    Args:
        df (pandas.DataFrame): A DataFrame containing "token" and "pos" columns.

    Returns:
        pandas.DataFrame: A new DataFrame with additional columns for token and pos bigrams.
    """
    # Create token bigrams
    df["token_bigram"] = pd.Series(list(zip(df["token"].shift(), df["token"])))
    # Create pos bigrams
    df["pos_bigram"] = pd.Series(list(zip(df["POS"].shift(), df["POS"])))
    return df

def read_data(file_path: str, save_to_csv: bool = False, output_file: str = None) -> pd.DataFrame:
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
    # for doc_i, doc in enumerate(train_data.split('\n\n')[:100]):
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
    named_enities_dict = {}
    for sent_id, group in train_df.groupby('sent_id'):
        sentence_tokens = list(group['token'])
        sentence = ' '.join(sentence_tokens)
        ne = get_named_entities(sentence)
        named_enities_dict[sent_id] = ne
    
                
    train_df['ner'] = train_df.apply(lambda x: assign_ne(x, named_enities_dict[x['sent_id']]), axis=1)
    train_df = add_bigrams(train_df)  
    # Convert the DataFrame from wide to long format
    df = train_df.melt(
        id_vars=[i for i in train_df.columns[:12]]+['ner','token_bigram','pos_bigram'], 
        var_name="notneeded", 
        value_name="target"
    )
    
    # Drop the 'notneeded' column and any rows that contain missing values
    df.drop(['notneeded'], axis=1, inplace=True)
    df = df[df['target'].notna()]
    
    df = df[df["distance_head"] != "_"]
    df.reset_index(inplace=True)
    
    # Optionally save the resulting DataFrame to a CSV file
    if save_to_csv:
        if output_file:
            df.to_csv(output_file, sep='\t', index=False)
        else:
            df.to_csv('processed_conll.tsv', sep='\t', index=False)
    # Return the resulting DataFrame
    return df

def read_conll(data_path):
    """
    Reads CoNLL data from the specified path and returns the training and testing dataframes.

    Args:
    data_path: str containing the path to the CoNLL data

    Returns:
    train_df: pandas DataFrame containing the training data
    test_df: pandas DataFrame containing the testing data
    """
    
    train_file = data_path+'/en_ewt-up-train.conllu'
    test_file = data_path+'/en_ewt-up-test.conllu'
    train_df = read_data(train_file, save_to_csv=False)
    test_df = read_data(test_file, save_to_csv=False)
    
    print("CHECK Dimensions: ",train_df.shape,test_df.shape)
    
    return train_df, test_df

def encode_features(train_df, test_df):
    """
    Encodes the features and targets in the training and testing dataframes.

    Args:
    train_df: pandas DataFrame containing the training data
    test_df: pandas DataFrame containing the testing data

    Returns:
    X_train: pandas DataFrame containing the encoded training features
    y_train: pandas Series containing the encoded training targets
    X_test: pandas DataFrame containing the encoded testing features
    y_test: pandas Series containing the encoded testing targets
    """

    # Get vectorized tokens
    train_tokens, test_tokens = vectorize_hash_token(train_df.token, test_df.token)

    # Get numerical features
    cols_num = ['sent_id', 'token_id', 'distance_head']
    train_num = numerical_features(train_df, cols_num)
    test_num = numerical_features(test_df, cols_num)

    # Get categorical features
    cols_cat_features = ['POS', 'uni_POS', 'morph_type',
                         'dep_label', 'dep_rel', 'space', 'probbank', 'ner',
                         'token_bigram', 'pos_bigram']
    train_cat, test_cat = categorical_features(train_df, test_df, cols_cat_features)

    # Concatenate everything into X_train
    X_train = pd.concat([train_tokens, train_cat, train_num], axis=1)
    X_test = pd.concat([test_tokens, test_cat, test_num], axis=1)

    # Turn targets into categorical labels
    y_train, y_test = make_cat_label(train_df['target'], test_df['target'])

    return X_train, y_train, X_test, y_test


def evaluate(y_test, y_pred):
    """
    Evaluates the performance of a classifier using F1-score, precision, and recall.

    Args:
    y_test: numpy array or list of true labels
    y_pred: numpy array or list of predicted labels

    Returns:
    None, but prints the F1-score, precision, and recall to console.
    """

    # Compute the F1-score, precision, and recall
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    # Print the results to console
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    
    
#PREPROCESSING

def make_binary_label(target_col):
    """
    Converts the target column to a binary label column.

    Args:
        target_col (pandas.Series): A pandas Series representing a target column in a DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame representing a binary label column with the following columns:
            - label: 1 if the target is an argumnet (not "_" or "V"), 0 otherwise.
    """
    labels = [0 if target == "_" or target == "V" else 1 for target in target_col]
    df = pd.DataFrame({"label": labels})
    return df

# def downsample(tokens_train,N):
#   return tokens_train.drop(tokens_train[tokens_train['target']=="_"].sample(n=N).index)
def downsample(tokens_train, N):
    """
    Downsamples a DataFrame by randomly removing a specified number of rows where the target column is "_".

    Args:
        tokens_train (pandas.DataFrame): A pandas DataFrame representing a training dataset with a "target" column.
        N (int): The number of rows to remove.

    Returns:
        pandas.DataFrame: A downsampled DataFrame with the same columns as the input DataFrame.
    """
    mask = tokens_train["target"] == "_"
    to_remove = tokens_train[mask].sample(n=N).index
    downsampled = tokens_train.drop(to_remove)
    return downsampled

def numerical_features(df, cols):
    """
    Casts the specified columns of a pandas DataFrame to integers.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the columns to be cast.
        cols (list of str): A list of column names to be cast to integers.

    Returns:
        pandas.DataFrame: A DataFrame with the same columns as the input DataFrame, 
        but with the specified columns cast to integers.
    """
    num_features = df[cols].astype(int)
    return num_features

def categorical_features(train_df, test_df, cols, n_out_feature=10):
    """
    Apply the hashing trick to the categorical features.

    Args:
        train_df (pandas.DataFrame): Training data.
        test_df (pandas.DataFrame): Test data.
        cols (list): List of categorical feature columns.
        n_out_feature (int): Number of output features for hashing.

    Returns:
        tuple: Hashed features for training and test data.

    """

    # Get categorical features
    cat_features_train = train_df[cols].astype(str)
    cat_features_test = test_df[cols].astype(str)

    # Apply the hashing trick to the categorical features
    hasher = FeatureHasher(n_features=n_out_feature, input_type='string')
    hashed_features_train = hasher.transform(cat_features_train.values)
    hashed_features_test = hasher.transform(cat_features_test.values)
    # Convert the output to pandas DataFrame
    hashed_features_train = pd.DataFrame(
        hashed_features_train.toarray(),
        columns=["cat_"+str(i) for i in range(n_out_feature)]
    )
    hashed_features_test = pd.DataFrame(
        hashed_features_test.toarray(),
        columns=["cat_"+str(i) for i in range(n_out_feature)]
    )
    return hashed_features_train, hashed_features_test

def vectorize_hash_token(train_tokens, test_tokens, n_out_feature=65):
    """
    Convert tokens into numerical vectors using FeatureHasher
    Args:
    train_tokens: A list of token strings for training data
    test_tokens: A list of token strings for test data
    N_outfeature: Number of output features for FeatureHasher (default=65)

    Returns:
    A tuple containing the transformed train and test data as pandas DataFrames
    """

    # Initialize FeatureHasher with input type 'string' and number of output features
    hasher = FeatureHasher(n_features=n_out_feature, input_type='string')

    # Transform train and test tokens into numerical vectors
    X_train = hasher.fit_transform(train_tokens)
    X_test = hasher.transform(test_tokens)

    # Convert the resulting sparse matrix to a dense matrix and concatenate with the numerical features
    X_train = pd.DataFrame(X_train.toarray(), columns=["tok_" + str(i) for i in range(0, n_out_feature)])
    X_test = pd.DataFrame(X_test.toarray(), columns=["tok_" + str(i) for i in range(0, n_out_feature)])

    return X_train, X_test


def make_cat_label(train_target, test_target):
    """
    Encode categorical labels using LabelEncoder.

    Args:
        train_target (array-like): Training set targets.
        test_target (array-like): Test set targets.

    Returns:
        tuple: Encoded training set targets and encoded test set targets.

    """
    encoder = LabelEncoder() 
    y_all = np.concatenate([train_target, test_target], axis=0)
    # Fit the label encoder to the targets
    encoder.fit(y_all.astype(str))

    y_train = encoder.transform(train_target.astype(str))
    y_test = encoder.transform(test_target.astype(str))

    return y_train,y_test


#MODEL 1
def get_helper1(X_train, X_test, y_train, y_test):
    """
    Get helper data from the first model using logistic regression.
    
    This function takes both train and test because the model that gives the information is the also
    trained on the training as well. It is advised to use this at the end of preprocessing for the 
    first model to benefit from all the features.
    
    Args:
    - X_train: A pandas dataframe containing the training data.
    - X_test: A pandas dataframe containing the test data.
    - y_train: A pandas series containing the training target variable.
    - y_test: A pandas series containing the test target variable.
    
    Returns:
    - helper1_train: A numpy array of predicted values for the training data.
    - helper1_test: A numpy array of predicted values for the test data.
    """
    # Convert target variables to binary labels
    y_train = make_binary_label(y_train)
    y_test = make_binary_label(y_test)

    # Create logistic regression model
    log_regression = LogisticRegression(penalty='l2')

    # Fit the model using the training data
    log_regression.fit(X_train, y_train)

    # Use model to make predictions on training and test data
    helper1_train = log_regression.predict(X_train)
    helper1_test = log_regression.predict(X_test)

    return helper1_train, helper1_test

#EVALS
def logistic_reg(X_train, X_test, y_train, y_test):
    """
    Perform logistic regression classification on the input data.

    Args:
    - X_train: pandas DataFrame containing the training data features
    - X_test: pandas DataFrame containing the test data features
    - y_train: pandas Series containing the training data target variable
    - y_test: pandas Series containing the test data target variable

    Returns:
    - y_pred: pandas Series containing the predicted target variable values for the test data
    """

    # instantiate the model
    log_regression = LogisticRegression(penalty='l2')

    # fit the model using the training data
    log_regression.fit(X_train, y_train)

    # use model to make predictions on test data
    y_pred = log_regression.predict(X_test)

    return y_pred

def evaluate_helper1(pred, y_test):
    """
    Evaluates the performance of the helper1 feature using the F1 score and the number of arguments predicted as relevant.

    Args:
        pred: A 1D array of predicted labels.
        y_test: A 1D array of true labels.

    Returns:
        None
    """
    f1 = f1_score(y_test, pred)
    arg_pred = pred.sum()
    nr_arg = y_test.sum()
    print(f"Score for the helper1 feature is {f1=}. {int(arg_pred)} out of the {int(nr_arg)} args in test were predicted.")

def confusion_mtrx_binary(y_test, y_pred):
    """
    This function generates a confusion matrix for binary classification.

    Args:
    - y_test (array-like): The true binary labels.
    - y_pred (array-like): The predicted binary labels.

    Returns:
    - None.
    """
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')#annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.color_palette("tab10")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['non_arg', 'arg'])
    ax.yaxis.set_ticklabels(['non_arg', 'arg'])

    return None

def downsampling_study(train, test, start=100000, end=900000, step=50000):
    """
    Perform a downsampling study by iteratively downsampling the training set and evaluating the performance 
    of a logistic regression model on the test set.
    
    Args:
    train: pandas DataFrame containing the training data
    test: pandas DataFrame containing the test data
    start: int, starting number of rows to remove from the training set
    end: int, ending number of rows to remove from the training set
    step: int, step size for removing rows from the training set
    
    Returns:
    A tuple containing three lists: 
    1) a list of the number of samples remaining in the training set after downsampling,
    2) a list of f1 scores obtained from evaluating the logistic regression model on the test set, and
    3) a list of the number of true positive predictions made by the logistic regression model on the test set.
    """

    down_sample_size = [x for x in range(start, end, step)]
    down_sample_size.reverse()  # reverse to have increasing remaining size of train test

    f1_scores = []
    true_predicts = []
    n_samples = []
    for x in down_sample_size:
        print(f"{x=}")
        strain = train.copy()
        stest = test.copy()
        strain = downsample(strain, x)

        train_size = strain.shape[0]
        n_samples.append(train_size)
        print(f"{train_size=}")

        X_train = strain
        y_train = X_train.pop('label')
        X_test = stest
        y_test = X_test.pop('label')

        y_pred = logistic_reg(X_train, X_test, y_train, y_test)

        npred = y_pred.sum()
        true_predicts.append(npred)
        print(f"{npred=}")

        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        print(f"{f1=}")
        print("-------------------X----------------")

    return n_samples, f1_scores, true_predicts

def plot_downsampling_study(n_sample, var, var_type):
    """
    Plot the specified variable against the number of training instances.

    Args:
    n_sample (list): List of integers representing the number of training instances.
    var (list): List of values of the variable to be plotted.
    var_type (str): String representing the variable type.

    Returns:
    None
    """
    plt.plot(n_sample, var)
    plt.xlabel("Number of instances")
    plt.ylabel(var_type)
    plt.title(var_type + " on number of training instances")
    plt.xticks([x for x in range(100000, 1000000, 200000)])
    plt.rcParams['figure.figsize'] = [3.8, 3.8]
    plt.grid('on')
