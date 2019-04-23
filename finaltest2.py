from sklearn.neural_network import MLPClassifier
import warnings
import pandas as pd
import numpy as np
import pickle
import time
import gc
import unicodedata
import h5py
import chunk
import sys
import csv
import re


################################################### FILE READING #######################################################

def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


def file_read_unique(location):
    with open(location, "rb") as fp:
        un = pickle.load(fp)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            return un


def save_file(file, iteration_number):
    filename = 'results' + iteration_number + '.h5'
    file.to_hdf(filename, key='df', mode='w')
    del file  # allow df to be garbage collected


################################################ APPLYING FILTERS ######################################################


def filtered_t(array, members):
    uniques = members['msno'].unique()
    array = array[array['msno'].isin(uniques)]
    return array


def filtered_songs(songs, res):
    songs = songs[songs['song_id'].isin(res)]
    return songs


#################################################### DUMMIES ###########################################################


def dummies(array):
    array = pd.get_dummies(array, sparse=True)
    return array


def songs_dummiess(array, column_name, col):
    array = array.astype('category', categories=col)
    array = pd.get_dummies(array, prefix=column_name, sparse=True).astype('int')
    return array


def genre_dummies(genre, name, un_vals, separator):
    #print('making dummies', separator)
    if len(separator) < 1:
        new_df = pd.get_dummies(genre, prefix=name, sparse=True)
        new_df = new_df.T.reindex(un_vals).T.fillna(0)
        return new_df
    elif len(separator) == 1:
        new_df = genre.astype('category', categories=un_vals)
        new_df = new_df.str.split(pat=separator)
        new_df = pd.get_dummies(new_df.apply(pd.Series).stack(), prefix=name, sparse=True).sum(level=0)
        return new_df
    else:
        separator = '[' + separator + ']'
        new_df = pd.DataFrame()
        for index, row in genre.iteritems():
            if new_df.empty:
                new_list = [re.split(separator, row)]
                new_list = [item for sublist in new_list for item in sublist]
                new_df = pd.DataFrame([new_list])
            else:
                new_list = [re.split(separator, row)]
                new_list = [item for sublist in new_list for item in sublist]
                new_list = pd.Series(new_list)
                new_list = new_list.apply(str)
                new_list = new_list.str.upper()
                new_list = new_list.str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
                new_df = pd.concat([new_df, pd.DataFrame([new_list])], ignore_index=True)
        col = un_vals.values.tolist()
        new_df = pd.Series(new_df.values.tolist())
        new_df = pd.get_dummies(new_df.apply(pd.Series).stack(), prefix=name, sparse=True).sum(level=0)
        new_df = new_df.T.reindex(col).T.fillna(0)
        mask = new_df.all(axis='columns')
        mask = mask[mask].index.values
        new_df['Other'] = new_df.set_value(mask, 'Other', 1, takeable=False)
        new_df = new_df.reindex(sorted(new_df.columns), axis=1)
        return new_df

def language_dummies(language, name, values):
    new_df = language.astype('category', categories=values)
    new_df = pd.get_dummies(new_df, prefix=name, sparse=True).astype('int')
    return new_df

def songs_dummies(array, column_name, col):
    # 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language'
    array = array.fillna('UNKNOWN')
    if column_name == 'language':
        new_df = language_dummies(array, column_name, col)
        return new_df
    elif column_name == 'genre_ids':
        seperator = find_genre_sep(array)
        new_df = genre_dummies(array, column_name, col, seperator)
        return new_df
    else:
        array = pd.Series(unicodedata.normalize('NFKC', item) for item in array.to_list())
        array = array.str.upper()
        seperator = find_sep(array)
        new_df = genre_dummies(array, column_name, col, seperator)
        return new_df


##################################### SIMPLE FUNTIONS FOR SMALL PARTS OF THE FILES #####################################

# train, songs, song_id
def dictionary_making(array_to_append, array, column_to_drop):
    if len(array.index)>1:
        dictionary = array.set_index(column_to_drop).T.to_dict('list')
        array_list = pd.Series(array_to_append[column_to_drop].map(dictionary))
        col = list(array.columns.values)
        col.remove(column_to_drop)
        array_list = pd.DataFrame(array_list.values.tolist(), columns=col)
        array_to_append = array_to_append.drop([column_to_drop], axis=1)
        array_list = array_list.reset_index(drop=True)
        array_to_append = pd.concat([array_to_append.reset_index(drop=True), array_list.reset_index(drop=True)], axis=1)
    else:
        column_number = len(array.columns)
        array = array.drop([array.columns[0]], axis=1)
        array = array.T
        column_names = array.index
        array = np.array(array.values).flatten()
        array = pd.Series(array)
        array = array.repeat(len(array_to_append.index))
        array = array.values()
        array = np.reshape(array, (len(array_to_append), column_number))
        array_to_append[[column_names]] = array
        array_to_append = array_to_append.drop(array_to_append.columns[0], axis=1)
        print(array_to_append)
    return array_to_append


def loading_bar(i, argument):
    if type(argument) == bool:
        argument = ' 0%'
    else:
        argument = argument[:-4]
        argument = argument + i * '█'
        if (len(argument)) < 10:
            argument = argument + ' 0' + str((len(argument))) + '%'
        else:
            argument = argument + ' ' + str((len(argument))) + '%'
        # time.sleep(1)
    sys.stdout.write('\r' + argument)
    sys.stdout.flush()
    return argument

def find_sep(column):
    unique_chars = ''.join(set(''.join(column.unique())))
    unique_chars = list(unique_chars)
    valid_seperators = ['and', 'AND', 'And', '\t', '(', ')', '`', "'", '-', '=', '~', '!', '@', '#', '$', '%', '^', '&',
                        '*', '+', '[', ']', '{', '}', ';', ':', '"', '|', '<', ',', '.', '/', '<', '>', '?']
    unique_chars = ''.join(set(unique_chars).intersection(valid_seperators))
    return unique_chars

def find_genre_sep(column):
    unique_chars = '|'
    return unique_chars



def find_optimal_N(value_counts):
    N = 0
    i = 0
    if value_counts.ix[i] > 1500:
        return True
    else:
        while N + value_counts.ix[i] <= 1500:
            N = N + value_counts.ix[i]
            i = i + 1
    return i


############################################## MANIPULATION FUNCTIONS ##################################################


def songs_manipulation(songs, values_list, max_length):
    col_list = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    for i in range(len(col_list)):
        dummy = songs_dummies(songs[col_list[i]], col_list[i], values_list[i])
        songs = pd.concat([songs.reset_index(drop=True), dummy.reset_index(drop=True)], axis=1)
        del dummy
    songs = songs.drop(col_list, axis=1)
    songs['song_length'] = pd.to_numeric(songs['song_length'])
    songs['song_length'] = (songs['song_length']/max_length)
    return songs


################################################### TRAINING ###########################################################


def train_them(path, songs_path, unique_val_list, max_length):
    print("ok training time!")
    mlp = MLPClassifier(warm_start=True)
    train_path = path + 'train_with_members_sorted_by_song_id.h5'
    value_counts_of_song = pd.read_hdf(path + 'train_value_counts_song.h5')

    # 7377418 is simply the total number of elements in the train file
    # It's only used to calculate how much time is left for the algorithm to complete.
    length_of_train = 7377418
    train_prev = 0
    track_prev = 0
    i = 0
    while not value_counts_of_song.empty:
        start_time = time.time()
        N = find_optimal_N(value_counts_of_song)

        # N is the number of IDs that get read
        # If TRUE is returned, instead of number, it means that a single ID gets repeated too many times
        # In this case, it's calculated multiple times, one for each instance it appears in the algorithm

        if type(N) != bool:

            # This only applies if it's the last loop
            if (len(value_counts_of_song.index)) < N:
                # The total variable, is to notify the reading algorithm at which row to stop reading
                # Since this is the last loop, all the numbers of the series are added
                total = train_prev + value_counts_of_song.sum()
                # Empties the list, so it doesn't loop forever
                value_counts_of_song = []
            else:
                # Gets the number of rows where the first N song_ids are repeated.
                # Effectively, the 'sum(value_counts_of_song.head(N).to_list())' corresponds to the same song_ids,
                # That N does. Then appends it to where the algorithm started to read previously.
                # Therefore, 'train_prev' marks the row in the file where the chunk starts, and 'total' where it ends
                total = train_prev + sum(value_counts_of_song.head(N).to_list())
                # Deletes the song_ids that have already been read
                value_counts_of_song = value_counts_of_song[N:]

            # Partially reads the files
            train_chunk = chunk_read(train_path, train_prev, total)
            track = chunk_read(songs_path, track_prev, track_prev + N)

            # Adds the number of read song_ids to the previous starting point
            track_prev = track_prev + N
            # Also the new train_prev, is the last line train_chunk was read
            train_prev = total

            # Creates dummies, and also normalizes the values
            track = songs_manipulation(track, unique_val_list, max_length)
            # Some ids do of 'track' do not exist in 'local_train' (they do exist in 'local_test' however,
            # and thus, are not deleted, since they'll be used later.
            # For the time being, the extra song_ids, are simply filtered out
            unique_ids = np.array(track['song_id'].unique()).flatten()
            local_train = train_chunk[train_chunk['song_id'].isin(unique_ids)]

            # Deletes the df that was read, to release memory
            del train_chunk
            gc.collect()

            # Separates target values from main array
            target_vals = local_train['target']
            # Drops the uneeded column
            local_train = local_train.drop(['target'], axis=1)

            # Each song_id in the local_train file, gets replaced by its corresponding values(which are found in track)
            local_train = dictionary_making(local_train, track, 'song_id')

            # Track gets deleted, to free up memory
            del track

            # Any NaN values are filled up, just in case
            local_train = local_train.fillna(0.0)
            print(local_train.shape)
            #mlp.fit(local_train, target_vals)
            print(str((total * 100) // length_of_train) + '%')
            i = i + 1
            if (i % 35) == 0:
                iteration_number = str(total // length_of_train)
                testing(mlp, path, songs, unique_val_list, iteration_number, max_length)
            print(time.time() - start_time)
            start_time = time.time()
        # Now we enter a scenario, where a certain song_id gets repeated too many times in the train file.
        # This results in insufficient memory.
        # In this case, we make multiple 'local_train' variables, over and over, for a single song_id
        else:
            start_time = time.time()
            # Prints the reason it loops!
            repeating_times = value_counts_of_song.iloc[0]
            total = total + repeating_times
            print('This song_id gets repeated ' + str(repeating_times) + ' times!')

            # The 'track' df, will only consist of a single row this time!
            # Of course, we'll only calculate it once, and then, we'll loop in the train file
            track = chunk_read(songs_path, track_prev, track_prev + 1)
            track = songs_manipulation(track, unique_val_list, max_length)

            # Reads all the rows where the song_id gets repeated.
            # Then, it's sliced in chunks!
            train_with_all_ids = chunk_read(train_path, train_prev, total)

            for h, local_train in train_with_all_ids.groupby(np.arange(repeating_times) // 1500):
                # No need to check for unique ids this time, since only one is used anyway
                # Separates target values from main array
                target_vals = local_train['target']
                # Drops the uneeded column
                local_train = local_train.drop(['target'], axis=1)

                # Each song_id in the local_train file,
                # Gets replaced by its corresponding values(which are found in track)
                local_train = dictionary_making(local_train, track, 'song_id')

                # Any NaN values are filled up, just in case
                local_train = local_train.fillna(0.0)
                print(local_train.shape)
                #mlp.fit(local_train, target_vals)
                print(str((total * 100) // length_of_train) + '%')
                i = i + 1
                if (i % 35) == 0:
                    iteration_number = str(total // length_of_train)
                    testing(mlp, path, songs, unique_val_list, iteration_number, max_length)
                print(time.time() - start_time)
                start_time = time.time()

            # Only one song_id was used all this time, so only one row will be deleted!
            value_counts_of_song = value_counts_of_song[1:]
            # Initializing the next train_prev
            train_prev = total
            # Again, only one row was used, so we simply add one!
            track_prev = track_prev + 1

        del local_train
        gc.collect()

        print(total, (total * 100) // length_of_train, i)
    testing(mlp, songs, unique_val_list, 1491632, max_length)


def testing(mlp, path, songs_path, unique_val_list, iteration_number,max_length):
    test_path = path + 'test_with_members_sorted_by_song_id.h5'
    accuracy_test = file_read(path_prefix + 'accuracy_test.h5')

    # Number of song_ids in test file are: 2556790

    value_counts_of_song_test = pd.read_hdf(path + 'test_value_counts_song.h5')

    # 7377418 is simply the total number of elements in the train file
    # It's only used to calculate how much time is left for the algorithm to complete.
    length_of_test = 2556790
    test_prev = 0
    track_prev = 0

    while not value_counts_of_song_test.empty:

        N = find_optimal_N(value_counts_of_song_test)

        # N is the number of IDs that get read
        # If TRUE is returned, instead of number, it means that a single ID gets repeated too many times
        # In this case, it's calculated multiple times, one for each instance it appears in the algorithm

        if type(N) != bool:

            # This only applies if it's the last loop
            if (len(value_counts_of_song_test.index)) < N:
                # The total variable, is to notify the reading algorithm at which row to stop reading
                # Since this is the last loop, all the numbers of the series are added
                total = test_prev + value_counts_of_song_test.sum()
                # Empties the list, so it doesn't loop forever
                value_counts_of_song_test = []
            else:
                # Gets the number of rows where the first N song_ids are repeated.
                # Effectively, the 'sum(value_counts_of_song.head(N).to_list())' corresponds to the same song_ids,
                # That N does. Then appends it to where the algorithm started to read previously.
                # Therefore, 'train_prev' marks the row in the file where the chunk starts, and 'total' where it ends
                total = test_prev + sum(value_counts_of_song_test.head(N).to_list())
                # Deletes the song_ids that have already been read
                value_counts_of_song_test = value_counts_of_song_test[N:]

            # Partially reads the files
            test_chunk = chunk_read(test_path, test_prev, total)
            track = chunk_read(songs_path, track_prev, track_prev + N)

            # Adds the number of read song_ids to the previous starting point
            track_prev = track_prev + N
            # Also the new train_prev, is the last line train_chunk was read
            test_prev = total

            # Creates dummies, and also normalizes the values
            track = songs_manipulation(track, unique_val_list, max_length)
            # Filtering
            unique_ids = np.array(track['song_id'].unique()).flatten()
            local_test = test_chunk[test_chunk['song_id'].isin(unique_ids)]

            # Deletes the df that was read, to release memory
            del test_chunk
            gc.collect()

            # Each song_id in the local_test file, gets replaced by its corresponding values(which are found in track)
            local_test = dictionary_making(local_test, track, 'song_id')

            # Track gets deleted, to free up memory
            del track

            # Any NaN values are filled up, just in case
            local_test = local_test.fillna(0.0)
            ids = local_test['id']
            local_test = local_test.drop(['id'], axis=1)
            new_target = pd.DataFrame(mlp.predict(local_test), columns=['target'])
            print(local_test.shape)
            del local_test, unique_ids
            gc.collect()
            new_target['id'] = ids

            comparison = accuracy_test[accuracy_test['id'].isin(new_target['id'])]
            d = comparison.merge(new_target, on=['id', 'target'], how='inner')
            print('the correct percentage is: ' + str(((len(d.index) / len(comparison.index)) * 100)) + '%')
            print(str((100 * total) // length_of_test) + '%')

            if total // length_of_test == 0:
                file = new_target
            else:
                file = pd.concat([file, new_target], ignore_index=True)

        # Now we enter a senario, where a certain song_id gets repeated too many times in the train file.
        # This results in insufficient memory.
        # In this case, we make multiple 'local_train' variables, over and over, for a single song_id
        else:
            # Prints the reason it loops!
            repeating_times = value_counts_of_song_test.iloc[0]
            total = total + repeating_times
            print('This song_id gets repeated ' + str(repeating_times) + ' times!')

            # The 'track' df, will only consist of a single row this time!
            # Of course, we'll only calculate it once, and then, we'll loop in the train file
            track = chunk_read(songs_path, track_prev, track_prev + 1)
            track = songs_manipulation(track, unique_val_list, max_length)

            # Reads all the rows where the song_id gets repeated.
            # Then, it's sliced in chunks!
            test_with_all_ids = chunk_read(test_path, test_prev, total)

            for h, local_test in test_with_all_ids.groupby(np.arange(repeating_times) // 1000):
                # No need to check for unique ids this time, since only one is used anyway

                # Each song_id in the local_train file,
                # Gets replaced by its corresponding values(which are found in track)
                local_test = dictionary_making(local_test, track, 'song_id')

                # Any NaN values are filled up, just in case
                local_test = local_test.fillna(0.0)
                ids = local_test['id']
                local_test = local_test.drop(['id'], axis=1)
                new_target = pd.DataFrame(mlp.predict(local_test), columns=['target'])
                print(local_test.shape)
                del local_test, unique_ids
                gc.collect()
                new_target['id'] = ids

                comparison = accuracy_test[accuracy_test['id'].isin(new_target['id'])]
                d = comparison.merge(new_target, on=['id', 'target'], how='inner')
                print('the correct percentage is: ' + str(((len(d.index) / len(comparison.index)) * 100)) + '%')

                print(str((total * 100) // length_of_test) + '%')

            iteration_number = str(total // length_of_test)
            # Only one song_id was used all this time, so only one row will be deleted!
            value_counts_of_song_test = value_counts_of_song_test[1:]
            # Initializing the next train_prev
            test_prev = total
            # Again, only one row was used, so we simply add one!
            track_prev = track_prev + 1

            file = pd.concat([file, new_target], ignore_index=True)

        if ((total * 100) // length_of_test) % 25 == 0:
            save_file(file, iteration_number)

        print(str(total // length_of_test) + '%')
    save_file(file, iteration_number)


################################################### MAIN STARTS HERE ###################################################

warnings.filterwarnings('ignore')
start_time = time.time()

print('\n' + "Making some strings...")
path_prefix = '/home/lydia/PycharmProjects/untitled/currently using/'

max_length = 99996

print('\n' + "Songs file reading starting...")
songs = path_prefix + "filtered_songs.h5"

print('\n' + "Filters reading starting...")
unique = file_read_unique(path_prefix + 'unique_genre_ids_v2.txt')

train_them(path_prefix, songs, unique, max_length)

final_time = time.time() - start_time
print(final_time)






























def most_reocurring_genres(genres):
    # fills the NaN with 0
    genres = genres.fillna(0)
    # splits the genres of each row, and creates new columns
    # the number of columns is 8, because that's the maximum number of different genres a movie can have
    # if a movie has less than 8 different genres, the extra columns are filed with None
    genres = genres.str.split(pat='|', expand=True)
    # creates an empty dataframe, in order to append values later on
    genres_pop = pd.DataFrame()
    # loops through each column
    for i in range(8):
        # appends VERTICALLY, how many times a value appears in each column
        genres_pop = genres_pop.append(genres.ix[:, i].value_counts().T)
    # there is need to sum up the values of each row. Therefore, all the NaN values are converted to 0
    genres_pop = genres_pop.fillna(0)
    # sums every row
    # now each row equates to a movie id, and how many times it appears, totally
    genres_pop = genres_pop.sum(axis=0)
    # normalizes
    genres_pop = (genres_pop - genres_pop.min()) / (genres_pop.max() - genres_pop.min())
    # converts to dict, so that the values can be mapped into the original genres column
    genres_pop = genres_pop.to_dict()
    # maps the values for each column
    for column in genres:
        genres.ix[:, column] = genres.ix[:, column].map(genres_pop)
    # keeps only the highest value
    genres = genres.max(axis=1)
    return genres


# counts how many times each value appears in the df
def most_reoccuring_people(array):
    indexes = array.value_counts()
    indexes = (indexes - indexes.min()) / (indexes.max() - indexes.min())
    indexes = indexes.to_dict()
    array = array.map(indexes)
    return array

def dictionary_makinnnng(array_to_append, array, column_to_drop):
    dictionary = array.set_index(column_to_drop).T.to_dict('list')
    array_list = pd.Series(array_to_append[column_to_drop].map(dictionary))
    col = list(array.columns.values)
    col.remove(column_to_drop)
    array_list = pd.DataFrame(array_list.values.tolist(), columns=col)
    # so far so good
    print(len(array_to_append.index), len(array_list.index))
    array_to_append = array_to_append.drop([column_to_drop], axis=1)
    array_to_append = pd.concat([array_to_append, array_list], axis=1)
    print(array_to_append)
    return array_to_append


'''elif(column=='hgfhg'):
            new_df = pd.Series(new_df.unique())
            new_df = new_df.str.split(pat='|', expand=True)
            un = []
            for new_col in new_df:
                inception = new_df[new_col].str.split(pat='/', expand=True)
                for col in inception:
                    inception[col] = inception[col].dropna()
                    n = inception[col].value_counts().T
                    mask = n >= n.size // 25
                    biggest = n[mask]
                    smallest = n[~mask]
                    biggest = list(biggest.index.values)
                    smallest = list(smallest.index.values)
                    # DO THIS WITH A MAP
                    n_dict = {}
                    for value in biggest:
                        n_dict[value] = value
                    for value in smallest:
                        n_dict[value] = 'Other'
                    inception[col] = inception[col].map(n_dict)
                    un = un + list(inception[col].unique())'''