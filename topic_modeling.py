# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 09:06:06 2016

@author: DIP
"""


from normalization import normalize_corpus
from utils import build_feature_matrix
import numpy as np
import os
from sklearn.decomposition import (LatentDirichletAllocation, NMF)


def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) 
                           for row 
                           in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) 
                               for wt, index 
                               in zip(weights,sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) 
                             for row 
                             in sorted_indices])
    
    topics = [np.vstack((terms.T, 
                     term_weights.T)).T 
              for terms, term_weights 
              in zip(sorted_terms, sorted_weights)]     
    
    return topics            
  
                       
def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):
    
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt,2)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
                     
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print


def topic_extraction(documents, labels):
    '''
    '''
    num_clusters = len(set(labels))
    n_topics = int(os.getenv('TOPIC_NUMBER_PER_CLUSTER'))
    matched = False
    for c in range(num_clusters):
        print("="*70)
        print("Cluster #{}:".format(c))
        corpus = [document for i,document 
                  in enumerate(documents) if labels[i] == c]
        norm_corpus = normalize_corpus(corpus)
        vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, 
                                                        feature_type='tfidf') 
        feature_names = vectorizer.get_feature_names()
        if os.getenv('TOPIC_MODELING') == "lda":
            lda = LatentDirichletAllocation(n_components=n_topics, 
                                            max_iter=1000,
                                            learning_method='online', 
                                            learning_offset=10.,
                                            random_state=42)
            lda.fit(tfidf_matrix)
            weights = lda.components_
            matched = True
            
        if os.getenv('TOPIC_MODELING') == "nmf":
            nmf = NMF(n_components=n_topics, 
                      random_state=42, alpha=.1, l1_ratio=.5)
            nmf.fit(tfidf_matrix)      
            weights = nmf.components_
            matched = True
        
        if not matched:
            raise ValueError("Unknown topic modeling algorithm!")
        
        topics = get_topics_terms_weights(weights, feature_names)
        print_topics_udf(topics=topics, 
                         total_topics=n_topics,
                         num_terms=10,
                         display_weights=True)

        
        