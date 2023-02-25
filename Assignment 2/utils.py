import spacy

nlp = spacy.load('en_core_web_sm')


def recombine_sentences(df):
    """
    Recombine the tokens in a data frame into sentences, with one sentence per sent_id.

    Args:
        df (pandas.DataFrame): A data frame with columns 'sent_id' and 'token', where each row contains a token in a sentence.

    Returns:
        dict: A dictionary where the keys are the sent_ids and the values are the corresponding sentences.

    """
    # Initialize an empty dictionary to store the sentences
    sentences = {}

    # Group the data frame by sent_id, and iterate over the resulting groups
    for sent_id, group in df.groupby('sent_id').first():
        # Extract the token values for the current group and concatenate them into a single string
        sentence_tokens = list(group['token'])
        sentence = ' '.join(sentence_tokens)

        # Store the resulting sentence in the dictionary with the sent_id as the key
        sentences[sent_id] = sentence

    # Return the resulting dictionary of sentences
    return sentences

def get_named_entities(df):
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
    
    # Combine the tokens in the data frame into sentences
    sentences = recombine_sentences(df)
    
    # Load the English model
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')
    
    # Process the sentence
    doc = nlp(sentence)
    
    

    # Initialize an empty dictionary to store the named entities
    named_entities = {}

    # Iterate over the sentences in the data frame
    for sent_id, sentence in sentences.items():
        # Use spaCy to extract named entities from the sentence
        doc = nlp(sentence)
        named_entities_for_sentence = [(ent.text, ent.label_) for ent in doc.ents]

        # Store the list of named entities in the dictionary with the sent_id as the key
        named_entities[sent_id] = named_entities_for_sentence

    # Return the resulting dictionary of named entities
    return named_entities

