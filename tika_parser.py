# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:45:33 2020

@author: Oleg
"""


import tika
# Unfortunately, the latest tika versions result in errors when parsing PDF
# files. Hence, the need for the older reliable version
if tika.__version__ != "1.19":
    raise Exception("Tika version must be 1.19!")
from tika import parser


def tika_parser(file_path):
    '''
    Parse a PDF file with tika and extract file content (text)
    '''
    
    # List of files that tika cannot parse
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