import numpy as np
import pandas as pd
import re
import string
import os
import networkx as nx
import codecs
from os import path
from bs4 import BeautifulSoup
from string import digits
from nltk.stem.snowball import FrenchStemmer
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle
import emoji
nltk.download('wordnet')
import spacy

### Remove html tags and uris from contents
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def strip_tags_and_uris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ''

def clean(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
#     x = re.sub(r'ô','o',x)
#     x = re.sub(r'[éêè]','e',x)
    x = re.sub(r'([^a-zéèûôàêô])',' ',x)
    x = re.sub('\n', '',x)
    x = re.sub("(\s+.\s+)", ' ', x)#remove any single charecters hanging between 2 spaces

    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

stops = set(stopwords.words('french'))
def remove_stopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

def remove_digits(x):
    rm_digits = str.maketrans('', '', digits)
    res = x.translate(rm_digits)
    return res

def remove_non_alpha(x):
    filtered_words = [word for word in x.split() if word.isalpha()]
    return " ".join(filtered_words)

# nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
nlp = spacy.load('fr', disable=['parser', 'ner'])

nlp.max_length = 10000000
lemmas_to_keep = ['NOUN', 'PROPN', 'VERB', 'ADJ']
def lemmatize(x):
    doc = nlp(x)
    lemmas = ' '.join(token.lemma_ for token in doc if token.pos_ in lemmas_to_keep)
    return lemmas

def hash_tag(x):
    return len( [ x for x in x.split() if x.startswith('#') ])

def user_tag(x):
    return len( [ x for x in x.split() if x.startswith('@') ])

def numeric_count(x):
    return len( [ x for x in x.split() if x.isdigit() ])

def count_emoji(text):

    emoji_list = []
    for word in text:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return len(emoji_list)

def length_doc(x):
    return  len(x.split())  

def avg_word_length(t):
    words = t.split()
    return ( sum( len(word) for word in words ) / len(words))

def full_preprocessing(df, filename):
    try:
        return pd.read_csv(filename)
    except:
        text = df["text"]
        df['hashtags_count'] = text.map(hash_tag)
        df['users_tagged']= text.map(user_tag)
        df['numeric'] = text.map(numeric_count)
        df['emojis']=text.map(count_emoji)
        
       

        text = text.map(strip_tags_and_uris)
        text = text.map(clean)
        text = text.map(remove_non_alpha)
        text = text.map(remove_stopwords)
        text = text.map(lemmatize)
        
        df['length_of_doc'] = text.map(length_doc)
        idx=np.where(df['length_of_doc']==0)
        df=df.drop(df.index[idx])
        text = df["text"]
        df['avg_word_length']=text.map(avg_word_length)
        df["text"] = text
        
     
        
        df.to_csv(filename)
    return df

def preprocessing_except_lemm(text):
    text = strip_tags_and_uris(text)
    text = clean(text)
    text = remove_non_alpha(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text
