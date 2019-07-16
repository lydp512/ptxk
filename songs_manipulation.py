from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import warnings
import time
import gc
import sys


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


def make_dict(good_list, bad_list):
    bad_list = bad_list[~bad_list.isin(good_list)].dropna()
    good_dict = {value: value for value in good_list}
    bad_dict = {key: 'OTHER' for key in bad_list}
    new_dict = {**good_dict, **bad_dict}
    return new_dict


def find_percentile(df):
    relevant_columns = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    list_of_dicts = []
    for column in relevant_columns:
        array = df[column].str.split(pat="|", expand=True)
        total = pd.DataFrame()
        for extra_column in array:
            add = array[extra_column].value_counts()
            if total.empty:
                total = add
                total_genre = pd.Series(array[extra_column].unique())
            else:
                total = pd.concat([total, add], axis=1)
                unique = pd.Series(array[extra_column].unique())
                unique = unique[~unique.isin(total_genre)].dropna()
                total_genre.append(unique)
        total = total.sum(axis=1)
        total = total.sort_values(ascending=False)
        total = total[total > total.quantile(.95)]

        allowed_list = total.index.tolist()
        allowed_list.append('OTHER')

        dict_for_map = make_dict(allowed_list, total_genre)
        list_of_dicts.append(dict_for_map)
    return list_of_dicts


def make_dummies(array, dict_for_map):
    array = array.str.split(pat="|", expand=True)
    total = pd.DataFrame()
    '''for column in array:
        add = array[column].value_counts()
        if total.empty:
            total = add
            total_genre = pd.Series(array[column].unique())
        else:
            total = pd.concat([total, add], axis=1)
            unique = pd.Series(array[column].unique())
            unique= unique[~unique.isin(total_genre)].dropna()
            total_genre.append(unique)
    total = total.sum(axis=1)
    total = total.sort_values(ascending=False)
    total = total[total > total.quantile(.95)]

    allowed_list = total.index.tolist()
    allowed_list.append('OTHER')

    dict_for_map = make_dict(allowed_list, total_genre)'''

    for column in array:
        array[column] = array[column].map(dict_for_map)
    array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
    return array


def manipulate(location):
    # length of file is 2296834
    entire_df = pd.read_hdf(location, 'df')
    list_of_maps = find_percentile(entire_df)
    max_song_length = entire_df['song_length'].max()
    min_song_length = entire_df['song_length'].min()
    del entire_df
    prev = 1
    while prev + 50000 < 2296834:
        entire_file = chunk_read(location, prev, prev + 50000)
        ids = entire_file['song_id']
        song_length = pd.to_numeric(entire_file['song_length'].drop(entire_file['song_length'].index[0]))
        # TypeError: ufunc 'subtract' did not contain a loop with signature matching types dtype('<U21') dtype('<U21') dtype('<U21')
        song_length = (song_length - min_song_length) / (max_song_length - min_song_length)
        languages = entire_file['language']
        languages = pd.get_dummies(languages)
        entire_file = entire_file.drop(['song_id', 'song_length', 'language'], axis = 1)
        i=0
        for column in entire_file:
            column = make_dummies(entire_file[column],list_of_maps[i])
            entire_file = pd.concat([entire_file, column], axis= 1)
            i=i+1
        entire_file = pd.concat([entire_file, languages], axis= 1)
        entire_file = pd.concat([song_length, entire_file],axis=1)
        entire_file = pd.concat([ids, entire_file], axis=1)
        prev = prev + 50000
        print(entire_file)
        asdf



################################################### MAIN STARTS HERE ###################################################
#warnings.filterwarnings('ignore')
start_time = time.time()
warnings.filterwarnings('ignore')

print('Reading first....')
path = '/home/lydia/PycharmProjects/untitled/currently using/songs_not_sorted.h5'


manipulate(path)

final_time = time.time() - start_time


print('It\'s finally done at ', final_time//3600, 'hours')
