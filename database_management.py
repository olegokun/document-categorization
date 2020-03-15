# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:51:21 2020

@author: Oleg
"""


import sqlite3


def sqlite_entry(path, title, content):
    '''
    Write a document title, document content, and the current time to 
    an sqlite database residing in the same directory where this script is
    '''

    conn = sqlite3.connect(path)
    c = conn.cursor()
    # Check if a title is in the database
    # If yes, then don't write a duplicate
    c.execute("SELECT COUNT(*), "\
              "SUM(CASE WHEN title=? THEN 1 ELSE 0 END) AS sum FROM document_db", 
              [title])
    row_count, n_records = c.fetchone()
    if not row_count or not n_records:
        c.execute("INSERT INTO document_db (title, content, date)"\
                  " VALUES (?, ?, DATETIME('now'))", (title, content))
    conn.commit()


def get_titles_content(path, indices):
    '''
    Get titles and the content of all documents with specified indices
    '''

    conn = sqlite3.connect(path)
    c = conn.cursor()
    # Check if a title is in the database
    # If yes, then don't write a duplicate
    c.execute("SELECT title, content FROM document_db")
    results = c.fetchall()
    conn.commit()
    return [(result[0], result[1]) for i, result in enumerate(results) \
            if i in indices]