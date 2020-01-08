from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import warnings
import pickle
import time
import sys


############################################ SIMPLE READING/SAVING FUNCTIONS ###########################################
def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
    return df


def file_save(file, filename):
    file.to_hdf(filename, key='df', mode='a')
    del file


def save_list(file, filename):
    with open(filename, "wb") as fp:
        pickle.dump(file, fp)


def read_list(filename):
    with open(filename, "rb") as fp:
        file = pickle.load(fp)
    return file


################################### NOT USED ATM, BUT ARE NEEDED FOR THE SONGS FILE ####################################
# Look at github for comments
def make_dict(good_list, bad_list):
    bad_list = bad_list[~bad_list.isin(good_list)].dropna()
    good_dict = {value: value for value in good_list}
    bad_dict = {key: 'OTHER' for key in bad_list}
    new_dict = {**good_dict, **bad_dict}
    return new_dict


def find_percentile(df):
    relevant_columns = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    list_of_dicts = []
    list_of_names = []
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
        list_of_names.append(allowed_list)
    return list_of_dicts, list_of_names


def find_lists(array):
    cities = array['city'].unique()
    registration = array['registered_via'].unique()
    del array
    return cities, registration


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


def correct(prediction, true_values):
    check = np.equal(prediction, true_values)
    check = np.sum(check)
    check = (check*100)/len(prediction)
    return check


def loading_bar(i):
    if i == 0:
        i = '0%'
    else:
        i = (i*2000000)//7377418
        i = (i//4)*'█' + " " + str(i) + "%"
    time.sleep(1)
    sys.stdout.write('\r' + i)
    sys.stdout.flush()


def loading_bar_test(i, score):
    if i == 0:
        time.sleep(1)
        sys.stdout.write(' 0%')
        sys.stdout.flush()
    else:
        i = (i*2000000)//2556789
        i = (i//4)*'█' + " " + str(i) + "%"
        time.sleep(1)
        sys.stdout.write('\r' + i + ' While the score is ' + str(score) + " %")
        sys.stdout.flush()


############################################### MAKES THE SONGS DUMMIES ###############################################
def make_dummies(array, dict_for_map, names, column_name):
    # NaN values aren't allowed, so they're filled as 'OTHER' (such subcategory exists in every category)
    array = array.fillna('OTHER')
    # Many artists can be found in a single line, seperated by "|". Split them.
    array = array.str.split(pat="|", expand=True)
    # Maps the column. Every genre/artist/etc that's not on the top 95% of popularity gets masked as 'OTHER'
    for column in array:
        array[column] = array[column].map(dict_for_map)
    # Makes dummies
    array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
    # Creates more columns for dummy variables that are not available
    array = array.T.reindex(names).T.fillna(0)
    array.columns = [column_name + '_' + str(col) for col in array.columns]
    # Keeps them in alphabetical order, so that the columns aren't in random order each time
    array = array.reindex(sorted(array.columns), axis=1)
    return array


###################################### PREPARES THE SONG ARRAY FOR THE ALGORITHM #######################################
def manipulate_song(array, maping, listing, max_length):
    # Song_length is appended as a string. Convert to float, so that normalization is possible
    array['song_length'] = array['song_length'].astype(float)
    # Some songs are not provided with a duration. Fill as 0
    array['song_length'] = array['song_length'].fillna(0)
    # Since minimum length is 0, simply divide by max length
    array['song_length'] = array['song_length'].div(max_length)
    # Get language dummies (no two languages appear in a single column,
    # so the extra function is not needed for this column)
    language = pd.get_dummies(array['language'])
    language = language.T.reindex(listing[4]).T.fillna(0)
    language.columns = ['language_' + str(col) for col in language.columns]
    language = language.reindex(sorted(language.columns), axis=1)
    array = array.drop(['language'], axis=1)
    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    i = 0
    # Makes dummies for each "more demanding" column
    '''for column in column_list:
        column = make_dummies(array[column], maping[i], listing[i], column)
        array = pd.concat([array, column], axis=1)
        i = i + 1'''
    column = make_dummies(array['genre_ids'], maping[0], listing[0], 'genre_ids')
    # Drops the original columns, since now they're replaced with the dummy columns
    array = pd.concat([array, language], axis=1)
    array = array.drop(['genre_ids', 'artist_name', 'composer', 'lyricist'], axis=1)
    return array


def manipulate_member(array, cities, registration, min_time, max_time, min_year, max_year):
    #['msno', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date']
    '''city = pd.get_dummies(array['city'])
    city = city.T.reindex(cities).T.fillna(0)
    city.columns = ['city_' + str(col) for col in city.columns]
    city = city.reindex(sorted(city.columns), axis=1)

    registered = pd.get_dummies(array['registered_via'])
    registered = registered.T.reindex(registration).T.fillna(0)
    registered.columns = ['registered_via_' + str(col) for col in registered.columns]
    registered = registered.reindex(sorted(registered.columns), axis=1)'''

    gender = pd.get_dummies(array['gender'])
    gender.columns = ['gender_' + str(col) for col in gender.columns]
    gender = gender.reindex(sorted(gender.columns), axis=1)

    array['bd'] = age(array['bd'])

    array['registration_init_time'] = array['registration_init_time']-min_time/max_time - min_time

    array = array.drop(['city', 'gender', 'registered_via'], axis=1)
    # array = pd.concat([array, city], axis=1)
    array = pd.concat([array, gender], axis=1)
    # array = pd.concat([array, registered], axis=1)

    array['year'], array['month'], array['day'] = date_for_members(array['expiration_date'], min_year, max_year)
    array = array.drop(['expiration_date'], axis=1)
    return array


###################################### PREPARES THE TRAIN ARRAY FOR THE ALGORITHM ######################################
def manipulate_train(array, categories):
    # Separates the target from the array
    target = array['target']
    array = array.drop(['target'],axis=1)
    # These are the categories that need to get dummies
    relevant_categories = ['source_system_tab', 'source_screen_name', 'source_type']
    '''# Make the dummies happen
    for i in range(3):
        column = array[relevant_categories[i]]
        column = pd.get_dummies(column)
        column = column.T.reindex(categories[i]).T.fillna(0)
        column.columns = [relevant_categories[i] + '_' + str(col) for col in column.columns]
        column = column.reindex(sorted(column.columns), axis=1)
        array = pd.concat([array,column], axis=1)'''
    # Drop the original columns, like before
    array = array.drop(relevant_categories, axis=1)
    return array, target


###################################################### MERGES THEM #####################################################
def implement(array, song, members):
    # Rather simple. Gets rid of the ids, and then merges them together. Simple as.
    song = song.drop(['song_id'],axis=1)
    array = array.drop(['msno', 'song_id'],axis=1)
    members = members.drop(['msno'], axis=1)
    array = pd.concat([array,song],axis=1)
    array = pd.concat([array,members],axis=1)
    array = array.fillna(0)
    return array


##################################################### TRAINS THEM #####################################################
def train_them(train, members, songs, maping, listing, max_length, categories):
    prev = 0
    i = 0
    mlp = MLPClassifier(learning_rate='adaptive', warm_start=True)
    entire_members = file_read(members)
    city_list, registration_list = find_lists(entire_members)
    min_time, max_time, min_year, max_year = find_min_and_max(entire_members['registration_init_time'],
                                                              entire_members['expiration_date'])
    # Reads in 10k chunks
    # 400000 einai to max
    while prev + 200000 < 7377418:
        loading_bar(i)
        # Takes care of the arrays first. Must be encoded, normalized and ready to go
        new_member = manipulate_member(chunk_read(members, prev, prev + 20000),city_list,registration_list,
                                       min_time,max_time,min_year,max_year)
        new_song = manipulate_song(chunk_read(songs, prev, prev+20000), maping, listing, max_length)
        new_train, target = manipulate_train(chunk_read(train, prev, prev+20000),categories)
        new_train = implement(new_train, new_song, new_member)
        # Feeds them into the algorithm, it's good to go!
        mlp.fit(new_train, target)
        prev = prev + 20000
        i = i + 1
    # Runs one last time. Same as before, it's just a smaller chunk now.
    new_member = manipulate_member(chunk_read(members, prev, 7377418), city_list, registration_list,
                                   min_time, max_time, min_year, max_year)
    new_song = manipulate_song(chunk_read(songs, prev, 7377418), maping, listing, max_length)
    new_train, target = manipulate_train(chunk_read(train, prev, 7377418), categories)
    new_train = implement(new_train, new_song, new_member)
    mlp.fit(new_train, target)
    loading_bar(i)
    print('\n')
    test_them(mlp, maping, listing, max_length, categories, city_list, registration_list,
              min_time, max_time, min_year, max_year)
    # Done with the fitting! Time to put it to the test!


def test_them(mlp, maping, listing, max_length, categories, city_list, registration_list,
              min_time, max_time, min_year, max_year):
    test_path = '/home/lydia/PycharmProjects/untitled/currently using/test_with_target.h5'
    test_songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs_test.h5'
    test_members_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members_test.h5'
    prev = 0
    i = 0
    while prev+20000 < 2556789:
        test_members = manipulate_member(chunk_read(test_members_path, prev, prev+20000), city_list, registration_list,
                                         min_time, max_time, min_year, max_year)
        test_song = manipulate_song(chunk_read(test_songs_path,prev,prev+20000), maping, listing, max_length)
        test, target = manipulate_train(chunk_read(test_path,prev,prev+20000), categories)
        test = implement(test, test_song, test_members)
        test = test.drop(['id'], axis=1)
        target = target.reset_index(drop=True)
        nan_index = target[target.apply(np.isnan)]
        nan_index = nan_index.index.tolist()
        target = target.dropna()
        prediction = mlp.predict(test)
        prediction = np.delete(prediction, nan_index)
        score = correct(prediction, target)
        prediction = pd.DataFrame(prediction)
        file_save(prediction, 'Score.h5')
        prev = prev + 20000
        loading_bar_test(i, score)
        i = i + 1
    test_members = manipulate_member(chunk_read(test_members_path, prev, 2556789), city_list, registration_list,
                                     min_time, max_time, min_year, max_year)
    test_song = manipulate_song(chunk_read(test_songs_path, prev, 2556789), maping, listing, max_length)
    test, target = manipulate_train(chunk_read(test_path, prev, 2556789), categories)
    test = implement(test, test_song, test_members)
    test = test.drop(['id'], axis=1)
    target = target.reset_index(drop=True)
    nan_index = target[target.apply(np.isnan)]
    nan_index = nan_index.index.tolist()
    target = target.dropna()
    prediction = mlp.predict(test)
    prediction = np.delete(prediction, nan_index)
    score = correct(prediction, target)
    prediction = pd.DataFrame(prediction)
    file_save(prediction, 'Score.h5')
    loading_bar_test(i, score)



############################################## MAIN STARTS HERE ################################################
warnings.filterwarnings('ignore')

# Simply reads the files
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
file_path = '/home/lydia/PycharmProjects/untitled/currently using/'
members_path = file_path + 'repeated_members.h5'
songs_path = file_path + 'repeated_songs.h5'
percentile_map = read_list(file_path + 'percentile_map.txt')
categories = read_list(file_path + 'categories.txt')
max_song_length = read_list(file_path + 'max_song_length.txt')
percentile_list = read_list(file_path + 'percentile_list.txt')


# All read and good. Ready to go!
train_them(train_path, members_path, songs_path, percentile_map, percentile_list, max_song_length, categories)
