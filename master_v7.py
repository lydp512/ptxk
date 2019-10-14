from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import warnings
import pickle
import time
import sys


############################################ SIMPLE READING/SAVING FUNCTIONS ###########################################
from sklearn.neural_network import MLPClassifier


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


def read_list(filename):
    with open(filename, "rb") as fp:
        file = pickle.load(fp)
    return file


def save_list(file, filename):
    with open(filename, "wb") as fp:
        pickle.dump(file, fp)


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


#################################################### SIMPLE FUNCTIONS ##################################################
def find_unique_songs(array):
    un = []
    array = array[1:]
    array['song_length'] = array['song_length'].astype(float)
    max_song_length = array['song_length'].max()
    array = array.drop(['song_id', 'song_length'], axis=1)
    for column in array:
        if column!='language':
            new = array[column].str.split(pat="|", expand=True)
            total = pd.DataFrame()
            for extra_column in new:
                add = new[extra_column].value_counts()
                if total.empty:
                    total = add
                    total_genre = pd.Series(new[extra_column].unique())
                else:
                    total = pd.concat([total, add], axis=1)
                    unique = pd.Series(new[extra_column].unique())
                    unique = unique[~unique.isin(total_genre)].dropna()
                    total_genre.append(unique)
            total_genre = total_genre.dropna()
            un.append(total_genre)
        else:
            total = array[column].unique()
            total = pd.Series(total)
            total = total.dropna()
            un.append(total)
    return un, max_song_length


def find_min_and_max(time, imerominia):
    min_time = time.min()
    max_time = time.max()
    # year month day format
    year, month, day = date(imerominia)
    min_year = year.min()
    max_year = year.max()
    return min_time, max_time, min_year, max_year


def age(array):
    array = array.where(array < 18, 18)
    array = array.where(array > 80, 23)
    return array


def date(array):
    array = array.astype(str)

    year = array.str[0:4]
    year = year.astype(int)

    month = array.str[4:6]
    month = month.astype(int)

    day = array.str[6:8]
    day = day.astype(int)
    return year, month, day


def date_for_members(array, min_year, max_year):
    year, month, day = date(array)
    year = (year - min_year)/(max_year-min_year)
    month = (month - 1)/11
    day = (day -1)/30
    return year, month, day


############################################### MAKES THE SONGS DUMMIES ###############################################
def make_dummies(array, names, column_name):
    # NaN values aren't allowed, so they're filled as 'OTHER' (such subcategory exists in every category)
    array = array.fillna(0)
    # Many artists can be found in a single line, seperated by "|". Split them.
    array = array.str.split(pat="|", expand=True)
    array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
    # Creates more columns for dummy variables that are not available
    array = array.T.reindex(names).T.fillna(0)
    array.columns = [column_name + '_' + str(col) for col in array.columns]
    # Keeps them in alphabetical order, so that the columns aren't in random order each time
    array = array.reindex(sorted(array.columns), axis=1)
    return array


############################################### CALCULATE THE DISTANCES ################################################
def dice_distance(array, column, pop):
    array = array.fillna('UNKNOWN')
    if column == 'language':
        array = pd.get_dummies(array)
        array = array.T.reindex(list(pop.columns)).T.fillna(0)
        array = array.reindex(sorted(array.columns), axis=1)
    else:
        # Genre_ids should be treated differently than the other columns, since it only has numerical values (with the
        # exception of 'UNKNOWN', but this is an input given by the program)
        if column == 'genre_ids':
            array = array.str.split(pat=r'\t|[()＋：．，`、／\'\-=~!@#$%^&*+\[\]{};:"|<,./<>?\\\\]', expand=True)
        else:
            # Many strings are in wide text (for example John vs Ｊｏｈｎ). Therefore, normalization is necessary.
            array = array.str.normalize('NFKC')
            # Plenty of lines, include more than a single artist. This means, that in order to count how many times an
            # artist appears in the dataframe, the data must be split.
            # Unfortunately, simply using "|" as a separator, isn't sufficient.
            # There are many different ways in which data are separated with, that's why all those possible separators
            # are included. This, however, leads to some more problems. Bands or artists that have any of these
            # separators in their name, have their name split (e.g. AC/DC). The gain of splitting them up, however, is
            # greater than the loss. So a few names, are indeed, sacrificed.
            array = array.str.split(pat=r'and|AND|And|\t|[1234567890()`、/\'\-=~!@#$%^&*+\[\]{};:"|<,.<>?\\\\]',
                                    expand=True)
        # Genre_ids are just numbers, so doing all of this isn't necessary.
        if column != 'genre_ids':
            # "trims up" the names of the artists. Empty spaces are disregarded, everything is converted to uppercase
            # and artists that may have appeared twice in the data(e.g. john and JOHN) are summed up in a single column.
            for new_col in array:
                array[new_col] = array[new_col].str.upper()
                array[new_col] = array[new_col].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
                array[new_col] = array[new_col].str.strip()
                #array = array[array.apply(lambda x: len(x) >= 2)]
                array[new_col] = array[new_col].apply(lambda x: np.NaN if (len(str(x)) < 2) else x)
                ASFASDFASDFSDFASDFASF
                MEMORY ERROR 
            array = array.stack().str.get_dummies().sum(level=0)
            print(array)
            # array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
            # Creates more columns for dummy variables that are not available
            array = array.T.reindex(list(pop.columns)).T.fillna(0)
            array = array.reindex(sorted(array.columns), axis=1)
        else:
            array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
            # Creates more columns for dummy variables that are not available
            array = array.T.reindex(list(pop.columns)).T.fillna(0)
            array = array.reindex(sorted(array.columns), axis=1)

    dist = DistanceMetric.get_metric('dice')
    dist = dist.pairwise(array, pop)
    dist = pd.Series(np.squeeze(dist))
    print(dist)
    return dist


################################################ MANIPULATION ALGORITHMS ###############################################
def manipulate_song(array, dice, max_len):

    # Normalize the numbers first, starting with song length
    array['song_length'] = array['song_length'].astype(float)
    # Some songs are not provided with a duration. Fill as 0
    array['song_length'] = array['song_length'].fillna(0)
    # Song_length is appended as a string. Convert to float, so that normalization is possible
    array['song_length'] = array['song_length'].div(max_len)

    # Now the dice distance must be calculated
    column_list = ['artist_name', 'composer', 'lyricist', 'language']
    #'genre_ids',
    i = 0
    for column in column_list:
        array[column] = dice_distance(array[column], column, dice[1])
        i = i+1
    print(array, array.shape, list(array.columns))
    # Done! Return new array and continue.
    return array


###################################### PREPARES THE SONG ARRAY FOR THE ALGORITHM #######################################
def train(song, member, train, dice_song, dice_member):
    #member_entire = file_read(member)
    # ['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language'] song
    # ['msno', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date'] member
    # ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'target'] train
    dice_song = read_list(dice_song)
    for array in dice_song:
        array = array.reindex(sorted(array.columns), axis=1)
    dice_member = read_list(dice_member)
    for array in dice_member:
        array = array.reindex(sorted(array.columns), axis=1)
    prev = 0
    i = 0
    mlp = MLPClassifier(learning_rate='adaptive', warm_start=True)
    #min_time, max_time, min_year, max_year = find_min_and_max(member_entire['registration_init_time'],
                                                              #member_entire['expiration_date'])
    #del member_entire
    #max_song_length = file_read(songs_path)['song_length'][1:].astype(float).max()
    max_song_length = 4145345
    # Reads in 10k chunks
    # 400000 einai to max
    prev = 0
    while prev + 20000 < 7377418:
        song = manipulate_song(chunk_read(song, prev, prev+20000), dice_song, max_song_length)
        train_chunk = chunk_read(train, prev, prev+20000)
    chunk = chunk_read(songs_path, prev, prev + 1000)



    language = pd.get_dummies(chunk['language'])
    language = language.T.reindex(unique[4]).T.fillna(0)
    language.columns = ['language_' + str(col) for col in language.columns]
    language = language.reindex(sorted(language.columns), axis=1)
    language = dice_distance(language, perc_list[4], unique[4])
    chunk['language'] = language

    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    # Makes dummies for each "more demanding" column
    i = 0
    for column in column_list:
        new_col = make_dummies(chunk[column], unique[i], column)
        new_col = dice_distance(new_col, perc_list[i], unique[i])
        chunk[column] = new_col
        i = i + 1
    print(chunk)
    asdf

    return array

warnings.filterwarnings('ignore')
songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
members_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members.h5'
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
test_path = '/home/lydia/PycharmProjects/untitled/currently using/test_with_target.h5'
songs_dice_path = '/home/lydia/PycharmProjects/untitled/python files/dice_dist_songs_list_90.txt'
members_dice_path = '/home/lydia/PycharmProjects/untitled/python files/dice_dist_members_list_90.txt'


train(songs_path, members_path, train_path, songs_dice_path, members_dice_path)
