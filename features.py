import pandas as pd
import stanza
import spacy
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
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,ner, constituency')

    # Load the English model (in spaCy)
    nlp_spacy = spacy.load('en_core_web_sm')


    # Process the sentence
    doc = nlp(sentence)

    doc_spacy = nlp_spacy(sentence)

    
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
    chunk_list = []
    constituency_list = []

    # Bc1 - 1st step
    # find all head words in the data
    head_words = [token.text for token in doc_spacy if token.head == token]


    # Extract features for each token in the sentence
    for i, sent in enumerate(doc.sentences):

        # Extract the noun phrases from the parsed text
        noun_chunks = list(doc_spacy.noun_chunks)

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


            # For every token, find the respective noun chunks and store them in a list
            chunks_found = []
            for chunk in noun_chunks:
                if (word.text) in str(chunk):
                    chunks_found.append(chunk)
                    continue

            # add chunks to list
            if len(chunks_found)>0:
                chunk_list.append(chunks_found)
            else:
                chunk_list.append('-')


            # Bc1
            # Get full constituent starting from (the?) head word in a sent.
            # Note: every sentence seems to have 1 head-word --> the constituency tree of every sentence belongs to the head-word?
            constituencies = sent.constituency

            # add them to list
            if str(word.text) in head_words:
                constituency_list.append(constituencies)
            else:
                constituency_list.append('-')



    # Create the DataFrame
    df = pd.DataFrame({
        'token': tokens,
        'pos': pos_tags,
        'lemma': lemmas,
        'ner': ner_tags,
        'stemming': stemmed_tokens,
        'pos_bigram': pos_bigrams,
        'token_bigram': token_bigrams,
        'noun chunks': chunk_list,
        'head_word_constituencies': constituency_list
    })


    return df


sent = "We are all in the gutter, but some of us are looking at the stars"
#sent = "Barack Obama was born in Hawaii.  He was elected president in 2008."
#sent = "We get it. Learning the meaning of the many words that make up the English language can seem overwhelming. Take away the nerves and make it simple and easy to understand with the use of our sentence maker."
#sent = "An armed man walked into an Amish school, sent the boys outside and tied up and shot the girls, killing three of them walked into an Amish school"

basic_fe = extract_features(sent)
print(basic_fe)