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
import tika
# Unfortunately, the latest tika versions result in errors when parsing PDF
# files. Hence, the need for the older reliable version
if tika.__version__ != "1.19":
    raise Exception("Tika version must be 1.19!")
from tika import parser
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.')/'categorization.env'
load_dotenv(dotenv_path=env_path)
from normalization import parse_document, normalize_corpus
from keyphrase_extraction import get_top_bigrams, get_top_trigrams
from utils import build_feature_matrix
from collections import Counter

#cur_dir = os.path.dirname(__file__)
#db = os.path.join(cur_dir, 'summaries.sqlite')


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


def tika_parser(file_path):
    '''
    Parse a PDF file with tika and extract file content (text)
    '''
    
    corrupted_files = []
    
    # Create a PDF object for a file
    try:
        content = parser.from_file(file_path)
        if 'content' in content:
            text = content['content']
        else:
            corrupted_files.append(file_path)
            return ' '
    except UnicodeEncodeError:
        corrupted_files.append(file_path)
        return ' '
    
    # Convert to string
    text = str(text)
    # Escape any \ issues
    text = str(text).replace('\n', ' ').replace('"', '\\"') 
    # Replace the return characters with a space, thus creating one long string
    text = ' '.join(text.split())
        
    return text, corrupted_files


def sqlite_entry(path, title, summary):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO summary_db (title, summary, date)"\
    " VALUES (?, ?, DATETIME('now'))", (title, summary))
    conn.commit()
    conn.close()


def main():
    # Get a list of file names of all documents in a specified folder
    fnames = get_filenames()
    print("The total number of files: %g" % len(fnames))
    #
    titles = []
    documents = []
    for i, fname in enumerate(fnames):
        print("*"*70)
        print("File no.%d %s is being priocessed ..." % (i, os.path.basename(fname)))
        text, corrupted_files = tika_parser(fname)
        if text:  # ignore corrupted files
            # Pre-process the file content
            sentences = parse_document(text)
            norm_sentences = normalize_corpus(sentences)
            bigrams = get_top_bigrams(norm_sentences, top_n=int(os.getenv('TOP_N')))
            flattened_bigrams = ' '.join(' '.join(tokens) for tokens in bigrams)
            trigrams = get_top_trigrams(norm_sentences, top_n=int(os.getenv('TOP_N')))
            flattened_trigrams = ' '.join(' '.join(tokens) for tokens in trigrams)
            all_tokens = flattened_bigrams + " " + flattened_trigrams
            documents.append(all_tokens)
            # Keep only the file name without extension
            titles.append(os.path.basename(fname).strip(".pdf"))
            # Write a summary to a SQLite database
            #sqlite_entry(db, title, summary)
    
    vectorizer, feature_matrix = \
    build_feature_matrix(documents, feature_type='tfidf', 
                         min_df=0.0, max_df=1.0,
                         ngram_range=(1, 1))
    print(feature_matrix.shape)
    # Get feature names
    feature_names = vectorizer.get_feature_names()
    
    topn_features = int(os.getenv('FEATURE_NUMBER'))
    
    if os.getenv('CLUSTERING') == "affinity":
        from document_clustering import affinity_propagation
        from document_clustering import (generate_wordclouds,
                                         get_cluster_data, 
                                         print_cluster_data, 
                                         plot_clusters)
        
        # Get clusters using affinity propagation
        ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
        
        centroids = pd.DataFrame(ap_obj.cluster_centers_)
        centroids.columns = feature_names
        generate_wordclouds(centroids)
                
        data = pd.DataFrame({'Title': titles, 'Cluster': clusters})
        # Get the total number of documents per cluster
        c = Counter(clusters)   
        print(c.items())  
        # Get the total number of clusters
        total_clusters = len(c)
        print('Total number of clusters:', total_clusters)
        
        cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                        data=data,
                                        feature_names=feature_names,
                                        num_clusters=total_clusters,
                                        topn_features=topn_features)         

        print_cluster_data(cluster_data) 

        plot_clusters(num_clusters=total_clusters, 
                      feature_matrix=feature_matrix,
                      cluster_data=cluster_data, 
                      data=data,
                      plot_size=(16,8)) 
    
    if os.getenv('CLUSTERING') == "kmeans":
        from document_clustering import k_means
        from document_clustering import (generate_wordclouds,
                                         get_cluster_data, 
                                         print_cluster_data, 
                                         plot_clusters)
        
        num_clusters = int(os.getenv('CLUSTER_NUMBER'))
        km_obj, clusters = k_means(feature_matrix=feature_matrix, 
                                   num_clusters=num_clusters)
        
        centroids = pd.DataFrame(km_obj.cluster_centers_)
        centroids.columns = feature_names
        generate_wordclouds(centroids)
        
        data = pd.DataFrame({'Title': titles, 'Cluster': clusters})
        # Get the total number of documents per cluster
        c = Counter(clusters)   
        print(c.items())
        
        cluster_data = get_cluster_data(clustering_obj=km_obj,
                                        data=data,
                                        feature_names=feature_names,
                                        num_clusters=num_clusters,
                                        topn_features=topn_features)         

        print_cluster_data(cluster_data) 
        
        plot_clusters(num_clusters=num_clusters, 
                      feature_matrix=feature_matrix,
                      cluster_data=cluster_data, 
                      data=data,
                      plot_size=(16,8))
    
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

        
if __name__ == '__main__':
    main()