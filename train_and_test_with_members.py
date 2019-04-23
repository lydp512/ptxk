import warnings
import pandas as pd
import numpy as np
import pickle
import re


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
    array = pd.get_dummies(array)
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



#test, members, msno
def dictionary_making(array_to_append, array, column_to_drop, name_of_file):
    dictionary = array.set_index(column_to_drop).T.to_dict('list')
    array_list = pd.Series(array_to_append[column_to_drop].map(dictionary))
    col = list(array.columns.values)
    col.remove(column_to_drop)
    array_list = pd.DataFrame(array_list.values.tolist(), columns=col)
    array_to_append = array_to_append.drop([column_to_drop], axis=1)
    array_list = array_list.reset_index(drop=True)
    # array_to_append = pd.concat([array_to_append.reset_index(drop=True), array_list.reset_index(drop=True)], axis=1)
    array_to_append = array_to_append.join(array_list)
    array_to_append.to_hdf(name_of_file, key='df', mode='w')
    del array_to_append  # allow df to be garbage collected
    print("ok")


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
        else:
            un = []
            new_list = array[column].unique()
            for item in new_list:
                un.append(re.split(r'\t|[（＋：，`、／\-=~!@#$%^&*+\[\]{};:"|<,./<>?]', item))
            un = [item for sublist in un for item in sublist]
            un = [" ".join(item.split()) for item in un]
            un = pd.Series(un)
            un = un[un.apply(lambda x: len(x) > 2)]
            n = un.value_counts().T
            if(column=='artist_name'):
                mask = n > 3
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
                un = un.map(n_dict)
            else:
                mask = n > 10
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
                un = un.map(n_dict)
        un = un.dropna()
        un = pd.Series(un.unique())
        un = un.sort_values(ascending=True)
        un = un.reset_index(drop=True)
        list_of_unique.append(un)
    return list_of_unique


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


#################################################### OTHER FUNCTIONS ###################################################

def accuracy(train, test):
    # file which is used to see how many ids were predicted correctly

    train = train[['msno', 'song_id', 'target']]
    test = test[['msno', 'song_id', 'id']]

    un_train = train['msno'].unique()
    test = test[test['msno'].isin(un_train)]
    un_test = test['msno'].unique()
    train = train[train['msno'].isin(un_test)]

    un = train['song_id'].unique()
    test = test[test['song_id'].isin(un)]
    un_test = test['song_id'].unique()
    train = train[train['song_id'].isin(un_test)]

    train = train.sort_values(['msno', 'song_id'], ascending=[True, True])
    test = test.sort_values(['msno', 'song_id'], ascending=[True, True])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train = train[['target']]
    test = pd.concat([test, train], axis=1)
    array_to_append = test.drop(['msno', 'song_id'], axis=1)
    array_to_append = array_to_append.dropna()
    array_to_append.to_hdf('accuracy_test.h5', key='df', mode='w')
    del array_to_append  # allow df to be garbage collected


################################################### MAIN STARTS HERE ###################################################

warnings.filterwarnings('ignore')


print('\n' + "train and test file reading starting...")
train = file_read("train.h5")
test = file_read("test.h5")
train = t_manipulation(train)
test = t_manipulation(test)

accuracy(train,test)

print('\n' + "Members file reading starting...")
members = file_read("members.h5")
members = members_manipulation(members)
print('\n' + "Done with Members!")

dictionary_making(test, members, 'msno', 'test_with_members.h5')
dictionary_making(train, members, 'msno', 'train_with_members.h5')

print("Making some filters....")
un_songs_train = np.array(train['song_id'].unique()).flatten()
un_songs_test = np.array(test['song_id'].unique()).flatten()
res = np.concatenate((un_songs_train, un_songs_test), axis=0)
res = res.flatten()
# total IDs: 584719

print('\n' + "Songs file reading starting...")
# songs file first
songs = file_read("songs.h5")
songs = filtered_songs(songs, res)
songs.to_hdf('filtered_songs.h5', key='df', mode='w')

unique_val_list = find_unique(songs)

with open("unique_genre_ids.txt", "wb") as fp:
    pickle.dump(unique_val_list, fp)