# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:13:06 2020

@author: Oleg
"""

import sqlite3
import os
import glob
import pandas as pd
import re
from tika_parser import tika_parser
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.')/'categorization.env'
load_dotenv(dotenv_path=env_path)
from normalization import preprocess_text
from utils import build_feature_matrix
from database_management import sqlite_entry
import pickle


def get_filenames():
    '''
    Extract full names (path and name) of all PDF files under a given directory
    '''
    
    allPdfFiles = glob.glob(os.getenv("BOOK_PATH") + "/*.pdf")
    
#    # Add some files manually as they are located in sub-directories
#    extraFileNames = ["\\Better Deep Learning/better_deep_learning.pdf",
#                     "\\Deep Learning for Computer Vision/deep_learning_for_computer_vision.pdf",
#                     "\\Deep Learning for NLP/deep_learning_for_nlp.pdf",
#                     "\\Deep Learning for Time Series Forecasting/deep_learning_time_series_forecasting.pdf",
#                     "\\Deep Learning with Python/deep_learning_with_python.pdf",
#                     "\\Introduction to Time Series Forecasting with Python/time_series_forecasting_with_python.pdf",
#                     "\\Long Short-Term Memory Networks with Python/long_short_term_memory_networks_with_python.pdf",
#                     "\\Generative Adversarial Networks with Python/generative_adversarial_networks.pdf",
#                     "\\Imbalanced Classification with Python/imbalanced_classification_with_python.pdf"]
#    # Combine two list of names
#    allPdfFiles.extend([os.getenv("BOOK_PATH") + fileName for fileName in extraFileNames])
    
    # For each file, extract filename while ignoring its extension 
    files = [re.split(".pdf", file)[0] for file in allPdfFiles]
    # For each extracted name, extract core name while ignoring version information such as "_v7" or "_v11_MEAP"
    # All core names related to the same book become now the same, i.e., standardized
    names = [re.split("_v[1-9]{1,2}", file)[0] for file in files]
    # Get creation time for each file
    creationTime = [os.path.getctime(fileName) for fileName in allPdfFiles]
    
    # Create a data frame with all derived information about files
    df = pd.DataFrame({"fullname": allPdfFiles, "name": names, "date": creationTime})
    # Sort rows first name, then by date
    df.sort_values(by=["name", "date"], ascending=True, inplace=True)
    # Group files with the same core name together and extract only the last file in each group as the latest by date
    df = df.groupby(by=["name"], sort=True).last()
    # Extract all latest versions of each file
    allPdfFiles = df["fullname"].values.tolist()
    return allPdfFiles


def main():
    '''
    Main function of document categorization
    '''
    
    # Get a list of file names of all documents in a specified folder
    fnames = get_filenames()
    print("The total number of files: %g" % len(fnames))
    
    titles = []
    documents = []
    for i, fname in enumerate(fnames):
        print("*"*70)
        print("File no.%d %s is being priocessed ..." % (i, os.path.basename(fname)))
        text, corrupted_files = tika_parser(fname)
        if text:  # ignore corrupted files
            all_tokens = preprocess_text(text)
            # Append this list as the new content describing the original document
            documents.append(all_tokens)
            # Keep only the file name without extension
            title = os.path.basename(fname).strip(".pdf")
            # Append the document title
            titles.append(title)
            # Write the document title and content to an SQLite database
            sqlite_entry(db, title, all_tokens)
    
    # Extract features from documents
    vectorizer, feature_matrix = \
    build_feature_matrix(documents, feature_type='tfidf', 
                         min_df=0.0, max_df=1.0,
                         ngram_range=(1, 1))
    # Notice that 'feature_matrix' is normalized so that no extra normalization 
    # is required
    
    # Save vectorizer in a file in the current folder
    with open(os.getenv('VECTORIZER_PKL_FILENAME'), 'wb') as file:
        pickle.dump(vectorizer, file)
        
    print(feature_matrix.shape)
    # Get feature names
    feature_names = vectorizer.get_feature_names()
    # Get the number of top features describing each cluster centroid 
    topn_features = int(os.getenv('FEATURE_NUMBER'))
    
    matched = False
    
    if os.getenv('CLUSTERING') == "affinity":
        from document_clustering import (affinity_propagation,
                                         cluster_analysis)
        from topic_modeling import topic_extraction
        
        # Get clusters using affinity propagation
        ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
        cl_obj = ap_obj
        
        cluster_analysis(ap_obj, feature_names, titles, clusters, 
                         topn_features, feature_matrix)
        
        # Extract topics of each cluster 
        tm_obj = topic_extraction(documents, ap_obj.labels_)
        
        matched = True
    
    if os.getenv('CLUSTERING') == "kmeans":
        from document_clustering import (k_means, 
                                         cluster_analysis)
        from topic_modeling import topic_extraction
        
        # Get clusters using k-means
        num_clusters = int(os.getenv('CLUSTER_NUMBER'))
        km_obj, clusters = k_means(feature_matrix=feature_matrix, 
                                   num_clusters=num_clusters)
        cl_obj = km_obj
        
        cluster_analysis(km_obj, feature_names, titles, clusters,
                         topn_features, feature_matrix)
        
        # Extract topics of each cluster
        tm_obj = topic_extraction(documents, km_obj.labels_)
        matched = True
        
    if os.getenv('CLUSTERING') == "hierarchical":
        from document_clustering import (ward_hierarchical_clustering, 
                                         plot_hierarchical_clusters)
        
        data = pd.DataFrame({'Title': titles})
        # Build ward's linkage matrix    
        linkage_matrix = ward_hierarchical_clustering(feature_matrix)
        # Plot the dendrogram
        plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                                   data=data,
                                   figure_size=(8,10))
        matched = True
    
    if not matched:
        raise ValueError("Unknown clustering algorithm!")
        
    # Save clustering and topic modeling objects in files in the current folder
    with open(os.getenv('CLUSTERING_PKL_FILENAME'), 'wb') as file:
        pickle.dump(cl_obj, file)
    with open(os.getenv('TOPIC_MODELING_PKL_FILENAME'), 'wb') as file:
        pickle.dump(tm_obj, file)
        
        
if __name__ == '__main__':
    # Create an sqlite database  if it does not exists and start processing
    cur_dir = os.path.dirname(__file__)
    db = os.path.join(cur_dir, 'documents.sqlite')
    if not os.path.exists(db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute('CREATE TABLE document_db'\
                  ' (title TEXT, content TEXT, date TEXT)')
    
    main()
    
    try:
        conn.close()
    except NameError:
        pass
