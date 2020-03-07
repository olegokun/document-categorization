# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 19:33:32 2016

@author: DIP
"""

from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures


def get_top_bigrams(corpus, top_n=100):
    '''
    Most frequent bigram detection
    '''
    
    finder = BigramCollocationFinder.from_documents([item.split() for item in corpus])
    bigram_measures = BigramAssocMeasures()                                                
    return finder.nbest(bigram_measures.raw_freq, top_n)   


def get_top_trigrams(corpus, top_n=100):
    '''
    Most frequent tri-gram detection
    '''
    
    finder = TrigramCollocationFinder.from_documents([item.split() for item in corpus])
    trigram_measures = TrigramAssocMeasures()                                                
    return finder.nbest(trigram_measures.raw_freq, top_n)
