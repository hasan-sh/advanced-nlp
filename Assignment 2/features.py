import pandas as pd
import stanza
import spacy
from nltk.stem import SnowballStemmer


# Load the English model
nlp = spacy.load("en_core_web_md")

# Initialize the Snowball stemmer
stemmer = SnowballStemmer('english')

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
    # nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')
    
    # Process the sentence
    doc = nlp(sentence)
    
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

def full_constituent(node, head_word, ce=[]):
    """
    Recursively traverses a parse tree to find the full constituent starting from a given head node.
    
    Args:
    - node (spacy.syntax.Nonterminal): The head node from which to find the full constituent.
    - ce (list): A list to store the constituent elements.
    
    Returns:
    - tuple: The full constituent as a tuple of (spacy.syntax.Nonterminal, list) where the first element is the
      head node of the constituent and the second element is a list of the constituent elements.
    """
    # If node has no children, return node and ce
    if not node.children:
        return node, ce
    
    # Traverse children and append children with label in ['NP', 'VP', 'PP'] to ce
    for child in node.children:
        if child.label in ['NP', 'VP', 'PP']:
            if head_word in child.leaf_labels():
                ce.append(child.leaf_labels())
        full_constituent(child, head_word, ce)
    
    # Return the last child and ce
    return child, list(filter(lambda x: head_word in x, ce))


def lookParent(sent, tok, d, pospath):
    """
    Recursively looks for the parent of a given token in a sentence and keeps track of the path
    in terms of part-of-speech tags.

    Args:
        sent (object): A sentence object that contains a list of words.
        tok (object): A token object representing the current token being looked at.
        d (int): An integer representing the current depth in the tree traversal.
        pospath (list): A list of part-of-speech tags representing the path from the original token to the root.

    Returns:
        Tuple: A tuple of the current depth and the updated part-of-speech tag path.

    """
    # If the token has no head, return the current depth and part-of-speech tag path
    if tok.head == 0:
        pospath.append(tok.pos)
        return d, pospath

    # If the token has a head, recursively call the function with the parent token
    else:
        d += 1
        pospath.append(tok.pos)
        headword = sent.words[tok.head - 1]
        return lookParent(sent, headword, d, pospath)


    
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner,depparse,constituency')
def extract_dependency_features(text):
    """
    Extracts various dependency features from a given sentence using the Stanza library.

    Args:
        text (str): The input text to extract features from.

    Returns:
        pandas.DataFrame: A dataframe containing the extracted features, with one row per token in the sentence.

    Features:
        - head (str): The text of the head word of each token.
        - full_constituent (str): The full constituent starting from the head of each token.
        - dependents (str): The text of all dependents of each token, separated by commas.
        - dep_path (str): The dependency path from the root to the head of each token.
    """
    doc = nlp(text)
    
    tree = doc.sentences[0].constituency
    
    rows = []
    for sent in doc.sentences:
        for word in sent.words:
            row = {
                'head': sent.words[word.head - 1].text if word.head != 0 else "ROOT",
                'full_constituent': full_constituent(tree, word.text)[1] if word.head == 0 else [word.text],
                'dependents': "",
                'dep_path': "",
                'path_len': lookParent(sent, word, 0, [])[0],
                'dep_rel': word.deprel, # GET DEPEPDENCY RELATION WITH ITS HEAD
            }
            
            # Build the dependency path from the root to the head of the token
            if word.head == 0:
                row['dep_path'] = "ROOT"
            else:
                path = []
                curr_word = word
                while curr_word.head != 0:
                    path.append(curr_word.text)
                    curr_word = sent.words[curr_word.head - 1]
                path.append("ROOT")
                row['dep_path'] = " -> ".join(reversed(path))
            
            # Traverse the dependency tree to extract all dependents of the word
            dependents = []
            stack = [word]
            while stack:
                curr_word = stack.pop()
                for dep_word in sent.words:
                    if dep_word.head == curr_word.id:
                        dependents.append(dep_word.text)
                        stack.append(dep_word)
            row['dependents'] = ", ".join(dependents)
            
            rows.append(row)
    
    return pd.DataFrame(rows)