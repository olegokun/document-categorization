# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:16:22 2020

@author: Oleg
"""

import sqlite3
import os
import pandas as pd
from tika_parser import tika_parser
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.')/'categorization.env'
load_dotenv(dotenv_path=env_path)
from normalization import preprocess_text
from utils import build_feature_matrix
import pickle


def update(fname):
    '''
    '''
    
    text, _ = tika_parser(fname)
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


if __name__ == '__main__':
    # Create an sqlite database  if it does not exists and start processing
    cur_dir = os.path.dirname(__file__)
    db = os.path.join(cur_dir, 'documents.sqlite')
    if not os.path.exists(db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute('CREATE TABLE document_db'\
                  ' (title TEXT, content TEXT, date TEXT)')
    
    # Load a TF-IDF vectorizer, clustering and topic modeling objects
    
    main()
    try:
        conn.close()
    except NameError:
        pass
