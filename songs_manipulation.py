from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import warnings
import time
import gc
import sys

###################################################### FILE READING ######################################################
# Simply reads the files, either entirely, or in chunks
def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


def save(file):
    filename = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
    file.to_hdf(filename, key='df', mode='w')
    del file
    

###################################################### EASY FUNCTIONS ######################################################
# Makes a dictionary, where all the popular categories are represented by their own names, and the rest by an 'OTHER'
def make_dict(good_list, bad_list):
    # At first, the 'bad list' includes the good values, so it's seperated
    bad_list = bad_list[~bad_list.isin(good_list)].dropna()
    # Then simply makes dictinaries out of the two lists, and then combine the two dictionaries to one
    good_dict = {value: value for value in good_list}
    bad_dict = {key: 'OTHER' for key in bad_list}
    new_dict = {**good_dict, **bad_dict}
    return new_dict

# Calculates the what the top 95th percentile values of each category are
def find_percentile(df):
    # Not all columns get dummy values, so not all of them need to be taken under consideration.
    # Moreover, the 'language' category, only has a few distinct values, so it's not needed to find the most popular ones     
    relevant_columns = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    list_of_dicts = []
    list_of_names = []
    for column in relevant_columns:
        # There are columns, where multiple values are connected with "|"
        # I separate them first, then create a new column for each value
        array = df[column].str.split(pat="|", expand=True)
        total = pd.DataFrame()
        for extra_column in array:
            # Counts the values for each column
            add = array[extra_column].value_counts()
            if total.empty:
                total = add
                total_genre = pd.Series(array[extra_column].unique())
            else:
                total = pd.concat([total, add], axis=1)
                unique = pd.Series(array[extra_column].unique())
                unique = unique[~unique.isin(total_genre)].dropna()
                total_genre.append(unique)
        # sums them up and calculates the top 95%
        total = total.sum(axis=1)
        total = total.sort_values(ascending=False)
        total = total[total > total.quantile(.95)]
        
        # The top names are appended to a list, all other values are represented by 'OTHER'
        allowed_list = total.index.tolist()
        allowed_list.append('OTHER')
        
        # Turns that into a dictinary (will be used later to map the values of the df)
        dict_for_map = make_dict(allowed_list, total_genre)
        list_of_dicts.append(dict_for_map)
        list_of_names.append(allowed_list)
    return list_of_dicts, list_of_names


##################################################### DUMMY FUNCTIONS #####################################################
def make_dummies(array, dict_for_map, names, column_name):
    # First splits the array since many values may appear in a single row
    array = array.str.split(pat="|", expand=True)
    # Maps the array with the dictionary that was created
    for column in array:
        array[column] = array[column].map(dict_for_map)
    # Default categories are used
    array = array.astype('category', categories = names)
    # Finally get dummies
    array = pd.get_dummies(array.apply(pd.Series).stack(), prefix=column_name).sum(level=0)
    return array

# The process that leads to the dummy creation
def dummmy_process(array, max_song_length, list_of_maps, list_of_names, un_languages):
    # Normalizes the song length
    song_length = array['song_length'].astype(float)
    song_length = song_length.fillna(0)
    # min length is 0
    song_length = song_length.div(max_song_length)
    
    # Creates language dummy values
    languages = array['language']
    languages = languages.astype('category', categories=un_languages)
    languages = pd.get_dummies(languages)
    
    # Dtops the now unneeded columns
    entire_file = array.drop(['song_length', 'language'], axis=1)
    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    i = 0
    
    # Dummy creation for other columns
    for column in column_list:
        column = make_dummies(entire_file[column], list_of_maps[i], list_of_names[i],column)
        entire_file = pd.concat([entire_file, column], axis=1)
        i = i + 1
    
    # Adds the usefull ones and deletes the non-needed
    entire_file = pd.concat([entire_file, languages], axis=1)
    entire_file = pd.concat([song_length, entire_file], axis=1)
    entire_file = entire_file.drop(['genre_ids', 'artist_name', 'composer', 'lyricist'], axis=1)
    return entire_file


######################################################## PREPERATION ########################################################
def manipulate(location):
    # length of file is 2296834
    
    # Reads the df
    entire_df = pd.read_hdf(location, start= 1)
    # Calculates all unique languages, as well as max song length, and the percentile dict and names
    un_languages = entire_df['language'].unique()
    max_song_length = entire_df['song_length'].max()
    max_song_length = float(max_song_length)
    list_of_maps, list_of_names = find_percentile(entire_df)
    print('Percentile done!')
    del entire_df
    prev = 1
    
    # Can't fit the entire df with the dummies in memory(obviously), so it's read in chunks
    while prev + 30000 < 2296834:
        entire_file = chunk_read(location, prev, prev + 30000)
        # Creates dummies
        entire_file = dummmy_process(entire_file, max_song_length, list_of_maps, list_of_names, un_languages)
        prev = prev + 30000
        # Writes it down
        entire_file.to_hdf('songs_95th_percentile.h5', key='df', mode='a')
    
    # Repeats the previous process one last time
    entire_file = chunk_read(location, prev, 2296834)
    entire_file = dummmy_process(entire_file, max_song_length, list_of_maps, list_of_names, un_languages)
    entire_file.to_hdf('songs_95th_percentile.h5', key='df', mode='a')


################################################### MAIN STARTS HERE ###################################################
# Counts the time
start_time = time.time()
warnings.filterwarnings('ignore')

# The main algorithm starts
print('Reading first....')
path = '/home/lydia/PycharmProjects/untitled/currently using/songs.h5'
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'

songs = file_read(songs_path)
train = file_read(train_path)
songs = songs[1:]
songs = pd.merge(train, songs, how='left', on='song_id')
songs = songs.drop_duplicates()

manipulate(path)

save(songs)

# Prints how long did it take and it's done!
final_time = time.time() - start_time
print('It\'s finally done at ', final_time//3600, 'hours')
