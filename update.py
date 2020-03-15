# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:16:22 2020

@author: Oleg
"""


from optparse import OptionParser
import sqlite3
from database_management import sqlite_entry, get_titles_content
import os, sys
from tika_parser import tika_parser
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.')/'categorization.env'
load_dotenv(dotenv_path=env_path)
from normalization import preprocess_text
import pickle
import numpy as np
from document_clustering import print_updated_cluster_data
from topic_modeling import updated_topic_extraction


def main():
    '''
    Inference mode: Assign a new document to the nearest cluster, 
    update the cluster centroid, and perform topic modeling
    '''

    text, _ = tika_parser(opts.fname)
    if text:  # ignore corrupted files
        all_tokens = preprocess_text(text)
        # Keep only the file name without extension
        title = os.path.basename(opts.fname).strip(".pdf")
        # Write the document title and content to an SQLite database
        sqlite_entry(db, title, all_tokens)
        # Extract features
        feature_matrix = vectorizer.transform([all_tokens]).todense()
        # Find a cluster to assign the extracted feature vector to
        c = cl_obj.predict(feature_matrix)[0]
        # Get the number of documents in the cluster c
        n = sum(cl_obj.labels_ == c)
        # Update the cluster centroid
        centroid = cl_obj.cluster_centers_[c,:]
        # Convert numpy.matrix to numpy.ndarray
        feature_array = np.ravel(feature_matrix)
        updated_centroid = (feature_array + n*centroid) / (n + 1)
        cl_obj.cluster_centers_[c,:] = updated_centroid
        # Update cluster labels too
        cl_obj.labels_ = np.append(cl_obj.labels_, c)
        
        # Display updated cluster's characteristics
        results = print_updated_cluster_data(db, cl_obj, c, vectorizer, title)
        
        # Update topics of the updated cluster
        documents = [result[1] for result in results]
        documents.append(all_tokens)
        tm_obj[c] = updated_topic_extraction(documents, tm_obj[c], c)        

        # Save the updated cl_obj and tm_obj in files in the current folder
        with open(os.getenv('CLUSTERING_PKL_FILENAME'), 'wb') as file:
            pickle.dump(cl_obj, file)
        with open(os.getenv('TOPIC_MODELING_PKL_FILENAME'), 'wb') as file:
            pickle.dump(tm_obj, file)


if __name__ == '__main__':
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--fname",
                  dest="fname", type="str",
                  help="PDF file name.")
    (opts, args) = op.parse_args(sys.argv)
    print(args)
    if len(args) != 1:
        op.error("This function takes one argument.")
        sys.exit(1)

    # Create an sqlite database  if it does not exists and start processing
    cur_dir = os.path.dirname(__file__)
    db = os.path.join(cur_dir, 'documents.sqlite')
    if not os.path.exists(db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute('CREATE TABLE document_db'\
                  ' (title TEXT, content TEXT, date TEXT)')
    
    # Load a TF-IDF vectorizer, clustering and topic modeling objects
    with open(os.getenv('VECTORIZER_PKL_FILENAME'), 'rb') as file:
        vectorizer = pickle.load(file)
    with open(os.getenv('CLUSTERING_PKL_FILENAME'), 'rb') as file:
        cl_obj = pickle.load(file)
    with open(os.getenv('TOPIC_MODELING_PKL_FILENAME'), 'rb') as file:
        tm_obj = pickle.load(file)
        
    main()
    
    try:
        conn.close()
    except NameError:
        pass
