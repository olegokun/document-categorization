# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:45:10 2016

@author: DIP
"""

import re
import nltk
import string


stopword_list = nltk.corpus.stopwords.words('english')
pos_tag_list = ['JJ', 'JJS', 'NN', 'NNP', 'RB', 'RBR', 'RBS', 'VBG', 'VBN']


def remove_noisy_tokens(text):
    # Replaces the ASCII '�' symbol with '8'
    text = text.replace(u'\ufffd', '8')   
    # Removes all numbers and words including them. e.g., 'Year2000'
    text = re.sub(r"\S*\d\S*", "", text)
    # Removes email addresses
    text = re.sub(r"\S*@\S*\s?", "", text)
    # Removes URLs starting with 'https' and 'http'
    text = re.sub(r"http\S+", "", text)
    # Removes URLs starting with 'www'
    text = re.sub(r"www\S+", "", text)
    return text.strip()
    
    
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

   
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens) 
    return filtered_text


def remove_special_pos(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token,pos_tag in nltk.pos_tag(tokens) if pos_tag in pos_tag_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    
    normalized_corpus = []
    
    for text in corpus:
        text = remove_noisy_tokens(text)
        text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        text = remove_special_pos(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
            
    return normalized_corpus


def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
    
    