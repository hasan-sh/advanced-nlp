import pandas as pd
import stanza
from nltk.stem import SnowballStemmer


def extract_features(sentence):
    """
    Extracts various linguistic features from a given sentence using the Stanza library.

    Args:
        sentence (str): The input sentence to extract features from.

    Returns:
        pandas.DataFrame: A DataFrame of features extracted from the sentence, with the following columns:
            - token (str): The original form of each token in the sentence.
            - pos (str): The part-of-speech tag for each token.
            - lemma (str): The lemma of each token.
            - ner (str): The named entity label for each token (if any).
            - stemming (str): The stemmed form of each token (using the Snowball stemmer).
            - pos_bigram (str): A string representing the part-of-speech bigram for each token and its successor.
            - token_bigram (str): A string representing the token bigram for each token and its successor.
    """
    # Load the English model
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')
    
    # Process the sentence
    doc = nlp(sentence)
    
    # Initialize the Snowball stemmer
    stemmer = SnowballStemmer('english')
    
    # Initialize lists to store the feature values
    tokens = []
    pos_tags = []
    lemmas = []
    ner_tags = []
    stemmed_tokens = []
    pos_bigrams = []
    token_bigrams = []
    
    # Extract features for each token in the sentence
    for i, sent in enumerate(doc.sentences):
        for j, word in enumerate(sent.words):
            # Add token to list
            tokens.append(word.text)
            
            # Add part-of-speech tag to list
            pos_tags.append(word.upos)
            
            # Add lemma to list
            lemmas.append(word.lemma)
            
            # Add named entity label to list if it exists, otherwise add an empty string
            ner_tags.append(word.ner if hasattr(word, 'ner') else '')
            
            # Add stemmed form to list
            stemmed_tokens.append(stemmer.stem(word.text))
            
            # Add part-of-speech bigram to list
            if j < len(sent.words) - 1:
                pos_bigrams.append(f"{word.upos}_{sent.words[j+1].upos}")
            else:
                pos_bigrams.append('')
                
            # Add token bigram to list
            if j < len(sent.words) - 1:
                token_bigrams.append(f"{word.text}_{sent.words[j+1].text}")
            else:
                token_bigrams.append('')
    
    # Create the DataFrame
    df = pd.DataFrame({
        'token': tokens,
        'pos': pos_tags,
        'lemma': lemmas,
        'ner': ner_tags,
        'stemming': stemmed_tokens,
        'pos_bigram': pos_bigrams,
        'token_bigram': token_bigrams
    })
    
    return df
