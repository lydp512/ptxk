import pandas as pd
import numpy as np
import unicodedata
import pickle
import re


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def file_save(file, filename):
    file.to_hdf(filename, key='df', mode='w')
    del file


def save_list(file, filename):
    with open(filename, "wb") as fp:
        pickle.dump(file, fp)


def read_list(filename):
    with open(filename, "rb") as fp:
        file = pickle.load(fp)
    return file


def make_dict(good_list, bad_list):
    bad_list = bad_list[~bad_list.isin(good_list)].dropna()
    good_dict = {value: value for value in good_list}
    bad_dict = {key: 0 for key in bad_list}
    new_dict = {**good_dict, **bad_dict}
    return new_dict


def popular_row(array, unique):
    #vriskei values sto array pou dn uparxoun sto unique
    #to unique einai mia lista me dataframes
    popular_values = pd.Series(data=0, index=unique.tolist())
    print(array, popular_values)
    upper = [x.upper() for x in list(array.index.values)]
    popular_values.loc[upper] = 1
    popular_values = pd.DataFrame(popular_values)
    popular_values = popular_values.transpose()
    return popular_values


def count(array, column, un):
    array = array.fillna('Unknown')
    # multiple names of each row are expanded into lists
    if column == 'language':
        cn = array.value_counts()
        cn = cn[cn > cn.quantile(.90)]
        cn = popular_row(cn,un)
    else:
        if column == 'genre_ids':
            array = []
            #array = array.str.split(pat=r'\t|[()＋：，`、／\'\-=~!@#$%^&*+\[\]{};:"|<,./<>?\\\\]', expand=True)
        else:
            array = array.str.split(pat=r'and|AND|And|\t|[1234567890()＋：，`、／\'\-=~!@#$%^&*+\[\]{};:"|<,./<>?\\\\]',
                                    expand=True)
        total = pd.DataFrame()
        for extra_column in array:
            add = array[extra_column].value_counts()
            if total.empty:
                total = add
                total_genre = pd.Series(array[extra_column].unique())
            else:
                total = pd.concat([total, add], axis=1, sort=False)
                unique = pd.Series(array[extra_column].unique())
                unique = unique[~unique.isin(total_genre)].dropna()
                total_genre.append(unique)
            del array[extra_column]
        if column != 'genre_ids':
            total = total.fillna(0.0)
            total['names'] = total.index
            total['names'] = total['names'].apply(str)
            total['names'] = total['names'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
            total = total[total['names'].apply(lambda x: len(x) > 2)]
            total['names'] = total['names'].str.upper()
            total = total.set_index('names')
            total = total.groupby(total.index).sum()
        total = total.sum(axis=1)
        total = total[total > total.quantile(.90)]
        cn = popular_row(total,un)
    return cn


def find_percentile(df):
    relevant_columns = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    dice = []
    path = '/home/lydia/PycharmProjects/untitled/old uses/unique_genre_ids_v2.txt'
    unique = read_list(path)
    i = 0
    for column in relevant_columns:
        dice.append(count(df[column], column, unique[i]))
        i = i+1
    return dice


'''train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
trains = file_read(train_path)
categories = []
for column in ['source_system_tab', 'source_screen_name', 'source_type']:
    trains[column] = trains[column].fillna('Unknown')
    categories.append(trains[column].unique())
del trains
save_list(categories, 'categories.txt')'''


songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
songs = file_read(songs_path)
songs = songs[1:]
un_languages = songs['language'].unique()
list_of_dice = find_percentile(songs)
#percentile_list.append(un_languages)
print(list_of_dice)
#save_list(list_of_dice, 'dice_dist_list_90.txt')
#save_list(percentile_map, 'percentile_map_50_percent.txt')
