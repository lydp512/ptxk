import warnings
import pandas as pd
import numpy as np
import pickle
import unicodedata
import re


################################################### FILE READING #######################################################

def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def find_unique(array):
    array = array.fillna('Unknown')
    print('\n' + 'Finding unique values in the songs file.')
    list_of_unique = []
    names_of_columns = list(array.columns.values)
    names_of_columns.remove('song_id')
    names_of_columns.remove('song_length')
    for column in names_of_columns:
        # multiple names of each row are expanded into lists
        if(column=='language'):
            un = array[column].unique()
            un = pd.Series(un)
            un = pd.to_numeric(un, errors='coerce').astype(np.int32)
            un = pd.Series(un.unique())
        else:
            un = []
            new_list = array[column].unique()
            for item in new_list:
                item = unicodedata.normalize('NFKC', item)
                if(column!="genre_ids"):
                    un.append(
                        re.split(r'and|AND|And|\t|[1234567890()＋：，`、／\'\-=~!@#$%^&*+\[\]{};:"|<,./<>?\\\\]', item))
                else:
                    un.append(
                        re.split(r'\t|[()＋：，`、／\'\-=~!@#$%^&*+\[\]{};:"|<,./<>?\\\\]', item))
            un = [item for sublist in un for item in sublist]
            un = [" ".join(item.split()) for item in un]
            un = pd.Series(un)
            un = un[un.apply(lambda x: len(x) > 2)]
            un = un.apply(str)
            un = un.str.upper()
            un = un.str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
            un = pd.Series(un.unique())
        un = un.dropna()
        un = un.sort_values(ascending=True)
        un = un.reset_index(drop=True)
        list_of_unique.append(un)
    return list_of_unique


################################################### MAIN STARTS HERE ###################################################

warnings.filterwarnings('ignore')
print('\n' + "Songs file reading starting...")
songs = file_read("filtered_songs.h5")

unique_val_list = find_unique(songs)

with open("unique_genre_ids_v2.txt", "wb") as fp:
    pickle.dump(unique_val_list, fp)