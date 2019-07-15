from sklearn.neural_network import MLPClassifier
import warnings
import pandas as pd
import numpy as np
import time
import h5py
import chunk
import sys
import csv


################################################### FILE READING #######################################################

def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df

################################################ APPLYING FILTERS ######################################################


def filtered_t(array, members):
    uniques = members['msno'].unique()
    array=array[array['msno'].isin(uniques)]
    return array


def filtered_songs(songs, res):
    songs = songs[songs['song_id'].isin(res)]
    return songs

#################################################### DUMMIES ###########################################################


def dummies(array):
    array = pd.get_dummies(array, sparse=True)
    return array


def songs_dummies(array, column_name, col):
    array = array.astype('category', categories=col)
    array = pd.get_dummies(array, prefix=column_name, sparse=True).astype('int')
    return array


def genres_dummies(genres):
    # splits the genres of each row into lists
    genres = genres.str.split(pat='|', expand=False)
    genres = genres.fillna(0)
    # stacks them up, with multi-indexing
    # then adds all the values of each index together
    genres = pd.get_dummies(genres.apply(pd.Series).stack()).sum(level=0)
    name_list = list(genres.columns.values)
    return genres, name_list


##################################### SIMPLE FUNTIONS FOR SMALL PARTS OF THE FILES #####################################


def dates(array):
    year = []
    month = []
    day = []
    for row in array:
        row = str(row)
        year.append(row[:4])
        month.append(row[4:6])
        day.append(row[6:])
    year = pd.Series(map(int, year))
    year = (year - year.min()) / (year.max() - year.min())
    month = pd.Series(map(int, month))
    month = (month - month.min()) / (month.max() - month.min())
    day = pd.Series(map(int, day))
    day = (day - day.min()) / (day.max() - day.min())
    return year, month, day


def age(array):
    fakes = array.between(12, 90)
    filter = array[fakes.values]
    array = array.where(fakes == True, filter.mean())
    array = (array - array.min()) / (array.max() - array.min())
    return array


def find_unique(array):
    print('\n' + 'Finding unique values in the songs file.')
    bar = loading_bar(25, True)
    list_of_unique = []
    names_of_columns = list(array.columns.values)
    names_of_columns.remove('song_id')
    names_of_columns.remove('song_length')
    for column in names_of_columns:
        new_df = array[column].fillna('Unknown')
        # multiple names of each row are expanded into lists
        if(column=='language'):
            un = array[column].unique()
        else:
            new_df = new_df.str.split(pat='|', expand=True)
            un = []
            for new_col in new_df:
                new_df[new_col] = new_df[new_col].dropna()
                n = new_df[new_col].value_counts().T
                mask = n >= n.size // 50
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
                new_df[new_col] = new_df[new_col].map(n_dict)
                un = un + list(new_df[new_col].unique())
        un = np.array(un)
        un = un.flatten()
        un = pd.Series(un)
        un = un.dropna()
        un = un.unique()
        list_of_unique.append(un)
        bar = loading_bar(20, bar)
    return list_of_unique



#test, members, msno
def dictionary_making(array_to_append, array, column_to_drop):
    dictionary = array.set_index(column_to_drop).T.to_dict('list')
    array_list = pd.Series(array_to_append[column_to_drop].map(dictionary))
    col = list(array.columns.values)
    col.remove(column_to_drop)
    array_list = pd.DataFrame(array_list.values.tolist(), columns=col)
    array_to_append = array_to_append.drop([column_to_drop], axis=1)
    array_list = array_list.reset_index(drop=True)
    # array_to_append = pd.concat([array_to_append.reset_index(drop=True), array_list.reset_index(drop=True)], axis=1)
    array_to_append = array_to_append.join(array_list)
    return array_to_append


def loading_bar(i, argument):
    if type(argument) == bool:
        argument = ' 0%'
    else:
        argument = argument[:-4]
        argument = argument + i*'█'
        if(len(argument)) < 10:
            argument = argument + ' 0' + str((len(argument))) + '%'
        else:
            argument = argument + ' ' + str((len(argument))) + '%'
        time.sleep(1)
    sys.stdout.write('\r' + argument)
    sys.stdout.flush()
    return argument


############################################## MANIPULATION FUNCTIONS ##################################################


def t_manipulation(array):
    dummy = dummies(array['source_system_tab'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'source_system_tab_' + str(values)
        array[column_name] = dummy[values]

    dummy = dummies(array['source_screen_name'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'source_screen_name' + str(values)
        array[column_name] = dummy[values]

    dummy = dummies(array['source_type'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'source_type' + str(values)
        array[column_name] = dummy[values]

    array = array.drop(['source_system_tab', 'source_screen_name', 'source_type'], axis=1)

    return array


def members_manipulation(members):
    dummy = dummies(members['city'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'city_' + str(values)
        members[column_name] = dummy[values]
    dummy = dummies(members['gender'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'gender_' + str(values)
        members[column_name] = dummy[values]
    dummy = dummies(members['registered_via'])
    dummy_list = list(dummy.columns.values)
    for values in dummy_list:
        column_name = 'registration_' + str(values)
        members[column_name] = dummy[values]
    members['registration_init_time_year'], members['registration_init_time_month'], members[
        'registration_init_time_day'] = dates(members['registration_init_time'])
    members['expiration_date_year'], members['expiration_date_month'], members['expiration_date_day'] = dates(
        members['expiration_date'])
    members['bd'] = age(members['bd'])
    members = members.drop(['city', 'gender', 'registered_via', 'registration_init_time', 'expiration_date'], axis=1)
    return members


def songs_manipulation(songs, values_list):
    col_list = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    for i in range(5):
        dummy = songs_dummies(songs[col_list[i]], col_list[i], values_list[i])
        songs = pd.concat([songs.reset_index(drop=True), dummy.reset_index(drop=True)], axis=1)
    songs = songs.drop(col_list, axis=1)
    songs['song_length'] = pd.to_numeric(songs['song_length'])
    songs['song_length'] = (songs['song_length'] - songs['song_length'].min()) / (
            songs['song_length'].max() - songs['song_length'].min())
    return songs


################################################### TRAINING ###########################################################


def train_them(train, test, members, songs):
    # must get all unique vals in order for the arrays to be of the same size when trained
    # therefore, this is a list of lists, with each list, enclosing all unique values of songs
    unique_val_list = find_unique(songs)

    print("\n" + "Splitting the df in chunks...")
    # songs is now an array of dataframes
    # song = [x for i, x in songs.groupby(level=0, sort=False)]

    print('\n' + "Making the members dictionary, and mapping...This will only happen once")
    train = dictionary_making(train, members, 'msno')

    mlp = MLPClassifier(warm_start=True)

    i = 1

    print('\n' + "Ok time to train...")
    learning_bar = loading_bar(20, True)
    print('\n')
    mini_bar = loading_bar(20, True)
    for g, track in songs.groupby(np.arange(len(songs)) // 5000):
        print(track.shape)
        if i % 100 == 5:
            learning_bar = loading_bar(i, learning_bar)
            print('\n')
            print(testing(mlp, test, members, songs, unique_val_list))
            mini_bar = loading_bar(20, True)

        # time to filter
        track = songs_manipulation(track, unique_val_list)
        print(track.shape)
        unique_ids = np.array(track['song_id'].unique()).flatten()
        local_train = train[train['song_id'].isin(unique_ids)]

        # separates target values from main array

        target_vals=local_train['target']
        local_train = local_train.drop(['target'], axis=1)

        local_train = dictionary_making(local_train, track, 'song_id')
        print(local_train.iloc[5])
        print(local_train.shape)

        mlp.fit(local_train, target_vals)
        mini_bar = loading_bar(1, mini_bar)
        i = i+1


def testing(mlp, test, members, songs, unique_val_list):
    print('Making the dictionaries. Fist for members')
    test = dictionary_making(test, members, 'msno')
    i = 1
    test_bar = loading_bar(24, True)
    for h, track in test.groupby(np.arange(len(test)) // 1000):
        # time to filter
        track = songs_manipulation(track, unique_val_list)
        unique_ids = np.array(track['song_id'].unique()).flatten()
        local_test = test[test['song_id'].isin(unique_ids)]

        local_test = dictionary_making(local_test, track, 'song_id')
        local_test = local_test.fillna(0)

        print(local_test.iloc[5])

        print(mlp.predict(local_test))
        test_bar = loading_bar(2, test_bar)

        i = i + 1


################################################### MAIN STARTS HERE ###################################################

warnings.filterwarnings('ignore')


print('\n' + "train and test file reading starting...")
bar = loading_bar(25, True)
train = file_read("train.h5")
bar = loading_bar(20, bar)
test = file_read("test.h5")
bar = loading_bar(10, bar)
train = t_manipulation(train)
bar = loading_bar(20, bar)
test = t_manipulation(test)
bar = loading_bar(5, bar)


print('\n' + "Members file reading starting...")
members = file_read("members.h5")
bar = loading_bar(1, bar)
members = members_manipulation(members)
bar = loading_bar(1, bar)
print('\n' + "Done with Members!")


print("Making some filters....")
un_songs_train = np.array(train['song_id'].unique()).flatten()
bar = loading_bar(4, bar)
un_songs_test = np.array(test['song_id'].unique()).flatten()
bar = loading_bar(2, bar)
res = np.concatenate((un_songs_train, un_songs_test), axis=0)
res = res.flatten()
bar = loading_bar(2, bar)
# total IDs: 584719

print('\n' + "Songs file reading starting...")
# songs file first
songs = file_read("songs.h5")
bar = loading_bar(30, bar)
songs = filtered_songs(songs, res)
bar = loading_bar(5, bar)


train_them(train, test, members, songs)
