from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from keras.layers import Dense, Dropout
from sklearn.utils import class_weight
from keras.layers import concatenate
from keras.models import Sequential
from keras.layers import LeakyReLU
from sklearn import preprocessing
from keras.layers import Input
from keras.models import Model
from math import isnan
import pandas as pd
import numpy as np
import warnings
import pickle
import keras
import time
import sys
import re


############################################ SIMPLE READING/SAVING FUNCTIONS ###########################################
def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def file_save(file, filename):
    file.to_hdf(filename, key='df', mode='a')
    del file


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


#################################################### SIMPLE FUNCTIONS ##################################################
def find_min_and_max(wra, imerominia):
    min_time = wra.min()
    max_time = wra.max()
    # year month day format
    imerominia = imerominia.replace(imerominia.min(), imerominia.median())
    min_date = imerominia.min()
    max_date = imerominia.max()
    return min_time, max_time, min_date, max_date


def age(array):
    median_calc = array[array>=9]
    median_calc = median_calc[median_calc<=90]

    array = array.where(array < 10, median_calc.median())
    # This is because, statistically, younger people are more likely to use false, old ages.
    array = array.where(array > 90, median_calc.median())
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
    year = (year - min_year) / (max_year - min_year)
    month = (month - 1) / 11
    day = (day - 1) / 30
    return year, month, day


def replace_gender(array):
    array = array.str.upper()
    # Any values that are not Male or Female are categorized as 0.5
    unique_vals = list(set(array.unique()) - set(['MALE', 'FEMALE']))
    array = array.replace({'MALE': 0, 'FEMALE': 1})
    array = array.replace(unique_vals, 0.5)
    return array


def correct(prediction, true_values):
    check = np.equal(prediction, true_values)
    check = np.sum(check)
    check = (check * 100) / len(prediction)
    return check


def loading_bar(i, total):
    if i == 0:
        i = '0%'
    else:
        i = (i * 12000000) // 7377418
        i = (i // 4) * '█' + " " + str(i) + "%."
        i = i + ' Total time that passed training the model is: ' + str(total // 60) + ' mins'
    time.sleep(1)
    sys.stdout.write('\r' + i)
    sys.stdout.flush()


def string_manipulation(array):
    array = array.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))
    array = array.str.split(
        pat=r'and|feat|FEAT|Feat|AND|And|\t|[1234567890`、/\'\-=~!@#$%^&*+{};:"|<,.<>?\\\\]', expand=True)
    for new_col in array:
        array[new_col] = array[new_col].str.upper()
        array[new_col] = array[new_col].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
        array[new_col] = array[new_col].str.strip()
        array[new_col] = array[new_col].apply(lambda x: None if (len(str(x)) < 2) else x)
    return array


def important_vals_easy(array, target):
    un = pd.unique(array.values.ravel('K'))
    rare_vals = array.value_counts()
    rare_vals = rare_vals[rare_vals <= 200]
    rare_vals = rare_vals.index.values.tolist()
    array = array[~array.isin(rare_vals)]
    array = array.dropna(how='all')
    array = pd.get_dummies(array)
    indexes = array.index
    forest_target = target.ix[indexes]
    clf = RandomForestClassifier(max_depth=2, random_state=0, warm_start=True)
    clf.fit(array, forest_target)
    importance = pd.Series(clf.feature_importances_)
    file_save(importance, 'importance.h5')
    value_name = pd.Series(array.columns)
    merged = pd.concat([importance, value_name], axis=1, ignore_index=True)
    merged = merged.set_index(1)
    merged = merged[~(merged<0.01).all(axis=1)]
    merged = merged.index.values.tolist()
    non_important = list(set(un) - set(merged))
    merged = dict(zip(merged,merged))
    non_important = {k:'OTHER' for k in non_important}
    merged = {**merged, **non_important}
    return merged


def important_vals_hard(array, target):
    un = pd.unique(array.values.ravel('K'))
    length = len(array.index)
    split = int((2*int(length)//(3*(len(un) ** (1/2)))))
    while not array.empty:
        dummy_array = array[:split]
        array = array[split:]
        dummy_array = dummy_array.dropna(how='all')
        dummy_array = pd.get_dummies(dummy_array.apply(pd.Series).stack()).sum(level=0)
        indexes = dummy_array.index
        forest_target = target.ix[indexes]
        clf = RandomForestClassifier(max_depth=2, random_state=0, warm_start=True)
        clf.fit(dummy_array, forest_target)
    importance = pd.Series(clf.feature_importances_)
    file_save(importance, 'importance.h5')
    value_name = pd.Series(dummy_array.columns)
    merged = pd.concat([importance, value_name], axis=1, ignore_index=True)
    merged = merged.set_index(1)
    merged = merged[~(merged<0.01).all(axis=1)]
    merged = merged.index.values.tolist()
    non_important = list(set(un) - set(merged))
    merged = dict(zip(merged,merged))
    non_important = {k:'OTHER' for k in non_important}
    merged = {**merged, **non_important}
    return merged


def chunk_creator(array, target):
    i=0
    while True:
        i = i+64
        dense = array[0]
        sparse = array[1]
        yield [dense[i:i+19200], sparse[i:i+19200]], target[i:i+19200]


def loading_bar_test(i, total, score):
    if i == 0:
        time.sleep(1)
        sys.stdout.write(' 0%')
        sys.stdout.flush()
    else:
        i = (i * 12000000) // 2556789
        i = (i // 4) * '█' + ' ' + str(i) + "%. Total time that passed testing the model is: " + str(
            total // 60) + ' minutes'
        time.sleep(1)
        sys.stdout.write('\r' + str(i) + ' While the score is ' + str(score) + " %")
        sys.stdout.flush()


############################################### CALCULATE THE DISTANCES ################################################
def fit_creation(entire_array, map_array, target):
    entire_array = entire_array.fillna('UNKNOWN')
    easy_cols = ['language', 'city', 'registered_via', 'source_system_tab', 'source_screen_name', 'source_type']
    harder_cols = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    for column in entire_array:
        array = entire_array[column]
        entire_array = entire_array.drop([column], axis=1)
        # The excluded columns have at least one separator, and therefore, must be treated differently
        if column in easy_cols:
            map = important_vals_easy(array,target)
            map_array.append(map)
        elif column in harder_cols:
            # Genre_ids should be treated differently than the other columns, since it only has numerical values (with the
            # exception of 'UNKNOWN', but this is an input given by the program)
            if column == 'genre_ids':
                array = array.str.split(pat='|', expand=True)
                map = important_vals_hard(array,target)
                map_array.append(map)
            else:
                # Many strings are in wide text (for example John vs Ｊｏｈｎ). Therefore, normalization is necessary.
                array = array.str.normalize('NFKC')
                rare_vals = array.value_counts()
                rare_vals = rare_vals[rare_vals <= 200]
                rare_vals = rare_vals.index.values.tolist()
                array = array[~array.isin(rare_vals)]
                # Plenty of lines, include more than a single artist. This means, that in order to count how many times an
                # artist appears in the dataframe, the data must be split.
                # Unfortunately, simply using "|" as a separator, isn't sufficient.
                # There are many different ways in which data are separated with, that's why all those possible separators
                # are included. This, however, leads to some more problems. Bands or artists that have any of these
                # separators in their name, have their name split (e.g. AC/DC). The gain of splitting them up, however, is
                # greater than the loss. So a few names, are indeed, sacrificed.
                array = string_manipulation(array)
                map = important_vals_hard(array, target)
                map_array.append(map)
    return map_array


def dummies(array, replace, i):
    array = array.str.normalize('NFKC')
    length = array.index
    array = string_manipulation(array)
    for column in array:
        array[column] = array[column].map(replace[i])
    array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
    array = array.reindex(length)
    array = array.drop(['OTHER'], axis=1)
    array = array.fillna(0)
    return array


################################################ MANIPULATION ALGORITHMS ###############################################
def manipulate_song(array, max_len, replace):
    array = array.fillna('UNKNOWN')
    # Normalize the numbers first, starting with song length
    array['song_length'] = array['song_length'].astype(float)
    # Some songs are not provided with a duration. Fill as 0
    array['song_length'] = array['song_length'].fillna(0)
    # Song_length is appended as a string. Convert to float, so that normalization is possible
    array['song_length'] = array['song_length'].div(max_len)


    genre_chunk = array['genre_ids'].str.split(pat='|', expand=True)
    for column in genre_chunk:
        genre_chunk[column] = genre_chunk[column].map(replace[0])
    genre_chunk = pd.get_dummies(genre_chunk.apply(pd.Series).stack()).sum(level=0)
    genre_chunk = genre_chunk.add_prefix('genre_ids_')
    array = array.drop(['genre_ids'], axis=1)
    array = pd.concat([array, genre_chunk], axis=1)

    lang_chunk = pd.get_dummies(array['language'].map(replace[4]))
    lang_chunk = lang_chunk.add_prefix('language_')
    array = array.drop(['language'], axis=1)
    array = pd.concat([array, lang_chunk], axis=1)


    # Now the dice distance must be calculated
    column_list = ['artist_name', 'composer', 'lyricist']
    i = 1
    for column in column_list:
        array[column] = array[column].str.normalize('NFKC')
        chunk = dummies(array[column], replace, i).add_prefix(column + '_')
        array = array.drop([column], axis=1)
        array = pd.concat([array, chunk], axis=1)
        i = i + 1
    # Done! Return new array and continue.
    return array


def manipulate_member(array, minT, maxT, minY, maxY, replace):
    # Convert to dice
    city_chunk = pd.get_dummies(array['city'].map(replace[5])).add_prefix('city_')
    reg_chunk = pd.get_dummies(array['registered_via'].map(replace[6])).add_prefix('registered_via_')
    array = array.drop(['city', 'registered_via'], axis=1)

    # Weed out possible fake values
    array['bd'] = age((array['bd']) - 10) / (90 - 10)

    # Converts male to 1, female to 0, and unknown/other to 0.5
    array['gender'] = replace_gender(array['gender'])

    # Normalizes time
    array['registration_init_time'] = ((array['registration_init_time'] - minT) / (maxT - minT))

    # Normalizes dates
    array['expiration_date'] = ((array['expiration_date'] - minY) / (maxY - minY))

    array = pd.concat([array, city_chunk, reg_chunk], axis=1)

    return array


def manipulate_t(array, song, member, replace):
    i = 7
    relevant_columns = ['source_system_tab', 'source_screen_name', 'source_type']
    for column in relevant_columns:
        chunk = pd.get_dummies(array[column].map(replace[i])).add_prefix(column + "_")
        array = array.drop([column], axis=1)
        array = pd.concat([array, chunk], axis=1)
        i = i + 1
    target = array['target']
    array = pd.concat([array, song, member], axis=1)
    array = array.drop(['target', 'msno', 'song_id'], axis=1)
    return array, target


def perceptron(perc, chunk, target):
    # xanaftiaxe ta data gt eisai zwo kai ta evales pali apo 0 mexri 1
    dense_cols = ['song_length', 'bd', 'gender', 'registration_init_time', 'expiration_date']
    dense_chunk = chunk[dense_cols]
    sparse_chunk = chunk.drop(dense_cols, axis=1)


    if perc == 0:

        # define two sets of inputs
        inputA = Input(shape=(5,))
        inputB = Input(shape=(109,))

        # the first branch operates on the first input
        x = Dense(5)(inputA)
        #x = Dropout(0.5)(x)
        #x = Dense(9, activation="relu")(x)
        #x = Dropout(0.5)(x)
        x = Dense(6, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(5, activation="relu")(x)
        x = Model(inputs=inputA, outputs=x)

        # the second branch opreates on the second input
        y = Dense(109, activation="relu")(inputB)
        #y = Dense(200, activation='relu')(y)
        #y = Dense(100, activation='relu')(y)
        #y = Dense(100, activation='relu')(y)
        y = Dense(50, activation='relu')(y)
        y = Dense(25, activation='relu')(y)
        #y = Dense(10, activation='softmax')(y)
        y = Model(inputs=inputB, outputs=y)

        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(30, activation="relu")(combined)
        #z = Dense(60, activation='relu')(z)
        #z = Dense(30, activation='relu')(z)
        z = Dense(8, activation='relu')(z)
        z = Dense(4, activation='relu')(z)
        z = Dense(1, activation='sigmoid')(z)
        #z = LeakyReLU(alpha=0.5)(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)

        x_train = dense_chunk.tail(750000)
        x_sparse = sparse_chunk.tail(750000)
        x_test = dense_chunk.head(50000)
        x_test_sparse = sparse_chunk.head(50000)
        y_train = target.tail(750000)
        y_test = target.head(50000)

        #class_weights = {0: 1.71682072, 1: 0.70545344}
        optimizer = keras.optimizers.Adagrad(learning_rate=0.025)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
        #model.fit([x_train,x_sparse], y_train, epochs=9, batch_size=64)

        for i in range(39):
            model.fit_generator(chunk_creator([x_train, x_sparse], y_train), steps_per_epoch=300, 
                                epochs=10, class_weight='balanced')
            print(model.evaluate([x_test, x_test_sparse], y_test, batch_size=256))
            s = pd.Series(np.squeeze(model.predict([x_test, x_sparse])))
            s = s.where(s > 0.8, 0)
            s = s.where(s < 0.8, 1)
            print(s)
            print(y_test)
            percent = s[s != y_test]
            percent = ((len(s.index) - len(percent.index))/len(s.index))*100
            print(percent)
        asdfasdf


###################################### PREPARES THE SONG ARRAY FOR THE ALGORITHM #######################################
def train(path_list):
    # ['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language'] song
    # ['msno', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date'] member
    # ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'target'] train


    '''i = 0
    mapped = []

    file = file_read(path_list[2])
    target = file['target']
    del file

    print("First, there needs to be done some work on the categorical data... This will take some time...")
    for path in path_list:
        file = file_read(path)
        mapped = fit_creation(file, mapped, target)
        if i == 1:
            min_time, max_time, min_date, max_date = find_min_and_max(file['registration_init_time'],
                                                                      file['expiration_date'])
        i = i+1
        del file
    save_list(mapped, 'map_file_fitted.txt')
    '''

    mapped = read_list('map_file_fitted.txt')
    i = 0
    for vl in mapped:
        mapped[i] = {k: v for k, v in vl.items() if k is not None}
        i = i+1

    #fitted = read_list('temp_file_fitted.txt')
    #placed = read_list('temp_file_placed.txt')
    file = file_read(path_list[1])
    min_time, max_time, min_year, max_year = find_min_and_max(file['registration_init_time'],
                                                              file['expiration_date'])
    del file

    max_song_length = 4145345
    prev = 0
    i = 0
    start = time.time()
    total_time = 0.0
    perc = 0
    while prev + 800000 < 7377418:
        loading_bar(i, total_time)
        '''song = manipulate_song(chunk_read(path_list[0], prev, prev+800000), max_song_length, mapped)
        member = manipulate_member(chunk_read(path_list[1], prev, prev+800000), min_time, max_time,
                                   min_year, max_year, mapped)
        train_chunk, target = manipulate_t(chunk_read(path_list[2], prev, prev + 800000), song,
                                           member, mapped)'''

        #file_save(train_chunk, 'just_for_test.h5')
        train_chunk = file_read('just_for_test.h5')
        target = file_read('just_for_test_target.h5')


        perc = perceptron(perc, train_chunk, target)
        total_time = time.time() - start
        prev = prev + 800000
        i = i + 1


warnings.filterwarnings('ignore')
songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
members_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members.h5'
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'

train([songs_path, members_path, train_path])
