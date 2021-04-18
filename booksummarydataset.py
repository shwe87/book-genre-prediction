# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import string
from torch.utils.data import Dataset


def clean(summary):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = summary.translate(table)
    text = re.sub(r"[^a-zA-Z0-9\s]"," ", summary.lower())
    return text


# Save the dictionary with its code.
VALID_GENRES = {
                #'Speculative fiction':0,
               'Science Fiction':0,
               'Crime Fiction':1,
               'Non-fiction':2,
               'Children\'s literature':3,
               'Fantasy':4,
               'Mystery':5,
               'Suspense':6,
               'Young adult literature':7}
                #'Speculative fiction':8,}
               #'Novel':9}

N_CLASSES = len(VALID_GENRES)
TARGETS = VALID_GENRES.keys()

#Function to get the genre from the code.
def get_genre(value):
    genres = list(VALID_GENRES.keys())
    codes = list(VALID_GENRES.values())
    position = codes.index(value)
    return genres[position]

class SummaryDataSet(Dataset):
    """The audio should be pre-processed before feeding it to the NN."""
    def __init__(self, filepath):
        self.book_summary_df = pd.read_csv(filepath)
        self.book_summary_df['Summary'] = self.book_summary_df['Summary'].map(lambda summary : clean(summary))   
    
    def __len__(self):
        return len(self.book_summary_df)

    def __getitem__(self, idx):
        summary = self.book_summary_df.iloc[idx]['Summary']
        genre = self.book_summary_df.iloc[idx]['Genres']
        return (
            summary,
            genre
        )