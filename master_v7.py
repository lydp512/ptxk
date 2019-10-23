from sklearn.neighbors.dist_metrics import DistanceMetric
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
    year, month, day = date(imerominia)
    min_year = year.min()
    max_year = year.max()
    return min_time, max_time, min_year, max_year


def age(array):
    array = array.where(array < 18, 18)
    # This is because, statistically, younger people are more likely to use false, old ages.
    array = array.where(array > 85, 18)
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


def replace_gender(array):
    array = array.str.upper()
    # Any values that are not Male or Female are categorized as 0.5
    unique_vals = list(set(array.unique()) - set(['MALE', 'FEMALE']))
    array = array.replace({'MALE': 1, 'FEMALE': 0})
    array = array.replace(unique_vals, 0.5)
    return array


def correct(prediction, true_values):
    check = np.equal(prediction, true_values)
    check = np.sum(check)
    check = (check*100)/len(prediction)
    return check


def loading_bar(i, total):
    if i == 0:
        i = '0%'
    else:
        i = (i*2000000)//7377418
        i = (i//4)*'█' + " " + str(i) + "%."
        i = i + ' Total time that passed training the model is: ' + total
    time.sleep(1)
    sys.stdout.write('\r' + i)
    sys.stdout.flush()


def loading_bar_test(i, total, score):
    if i == 0:
        time.sleep(1)
        sys.stdout.write(' 0%')
        sys.stdout.flush()
    else:
        i = (i*2000000)//2556789
        i = (i//4)*'█' + " " + str(i) + "%. Total time that passed testing the model is: " + total
        time.sleep(1)
        sys.stdout.write('\r' + i + ' While the score is ' + str(score) + " %")
        sys.stdout.flush()


############################################### CALCULATE THE DISTANCES ################################################
# HUGE amounts of RAM are saved, compared to just using the regular sklearn dice distance to start with, due to weeding
# out the true negatives before continuing.
def calc_dist(main_array, pop_values):
    # First, the true negatives are disregarded from the calculation
    columns_to_be_dropped = list(pop_values.loc[:, (pop_values == 0).all()].columns)
    # Since only values which exist at least once, are implemented in the main array, it means that the corresponding
    # list would include all the column names. Therefore, a simple list of all column names is enough.
    non_zero_lines = list(main_array.columns)
    columns_to_be_dropped = list(set(columns_to_be_dropped) - set(non_zero_lines))
    pop_values = pop_values.drop(columns=columns_to_be_dropped)
    # Now, append zeros in the main array, since it corresponds to non-zero values in the pop_values(basically what is
    # done, is to append the false negatives. Also, it reindexes the dataframe, to correspond to the pop_values values.
    main_array = main_array.T.reindex(list(pop_values.columns)).T.fillna(0)
    dist = DistanceMetric.get_metric('dice')
    # All good to go. Counts the distance, squeezes it back to a pandas Series, and it's all done!
    dist = dist.pairwise(main_array, pop_values)
    dist = pd.Series(np.squeeze(dist))
    return dist


def dice_distance(array, column, pop):
    array = array.fillna('UNKNOWN')
    # The excluded columns have at least one separator, and therefore, must be treated differently
    if column != 'genre_ids' and column != 'artist_name' and column != 'composer' and column != 'lyricist':
        array = pd.get_dummies(array)
        dist = calc_dist(array, pop)
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
                array[new_col] = array[new_col].apply(lambda x: None if (len(str(x)) < 2) else x)
            array = pd.get_dummies(array.stack()).sum(level=0)
            # During the dummy process, some columns that lack a value, get deleted completely.
            # The following line of code, makes sure to restore them.
            array = array.reindex(list(range(array.index.min(), array.index.max()+1)), fill_value=0)
            # No need to merge the arrays already. We need to weed out the true negatives first, since dice distance
            # is used. It is not only more memory and time efficient to not merge them yet, but the solution appears
            # much more clean as well.
            dist = calc_dist(array, pop)
        else:
            array = pd.get_dummies(array.apply(pd.Series).stack()).sum(level=0)
            # Creates more columns for dummy variables that are not available
            dist = calc_dist(array, pop)
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
    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    i = 0
    for column in column_list:
        array[column] = dice_distance(array[column], column, dice[i])
        i = i+1
    # Done! Return new array and continue.
    return array


def manipulate_member(array, dice_list, minT, maxT, minY, maxY):

    # Convert to dice
    array['city'] = dice_distance(array['city'], 'city', dice_list[0])
    array['registered_via'] = dice_distance((array['registered_via']), 'registered_via', dice_list[1])

    # Weed out possible fake values
    array['bd'] = age(array['bd'])

    # Converts male to 1, female to 0, and unknown/other to 0.5
    array['gender'] = replace_gender(array['gender'])

    # Normalizes time
    array['registration_init_time'] = array['registration_init_time'] - minT / maxT - minT

    # Normalizes dates
    array['year'], array['month'], array['day'] = date_for_members(array['expiration_date'], minY, maxY)
    array = array.drop(['expiration_date'], axis=1)

    return array


def manipulate_t(array, song, member, dice):
    i = 0
    relevant_columns = ['source_system_tab', 'source_screen_name', 'source_type']
    for column in relevant_columns:
        array[column] = dice_distance(array[column], column, dice[i])
    target = array['target']
    array = pd.concat([array, song, member], axis=1)
    array = array.drop(['target', 'msno', 'song_id'], axis=1)
    return array, target


###################################### PREPARES THE SONG ARRAY FOR THE ALGORITHM #######################################
def train(song_path, member_path, train_path, dice_song, dice_member, dice_train):
    # ['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language'] song
    # ['msno', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date'] member
    # ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'target'] train
    dice_song = read_list(dice_song)
    dice_member = read_list(dice_member)
    dice_train = read_list(dice_train)
    mlp = MLPClassifier(learning_rate='adaptive', warm_start=True)
    member_entire = file_read(member_path)
    min_time, max_time, min_year, max_year = find_min_and_max(member_entire['registration_init_time'],
                                                              member_entire['expiration_date'])
    del member_entire
    max_song_length = 4145345
    # Reads in 20k chunks
    prev = 0
    i = 0
    start = time.time()
    total_time = 0.0
    while prev + 30000 < 7377418:
        loading_bar(i, total_time)
        song_chunk = manipulate_song(chunk_read(song_path, prev, prev+30000), dice_song, max_song_length)
        member_chunk = manipulate_member(chunk_read(member_path, prev, prev+30000), dice_member, min_time, max_time,
                                         min_year, max_year)
        train_chunk, target = manipulate_t(chunk_read(train_path, prev, prev+30000), song_chunk,
                                               member_chunk, dice_train)
        # kapou exei NaN, ftiax'to
        mlp.fit(train_chunk, target)
        total_time = time.time() - start
        test(mlp, dice_song, dice_member, dice_train, max_song_length, min_time, max_time, min_year, max_year)
        prev = prev + 30000
    # Runs one final time
    song_chunk = manipulate_song(chunk_read(song_path, prev, 7377418), dice_song, max_song_length)
    member_chunk = manipulate_member(chunk_read(member_path, prev, 7377418), dice_member, min_time, max_time,
                                     min_year, max_year)
    train_chunk, target = manipulate_t(chunk_read(train_path, prev, 7377418), song_chunk,
                                           member_chunk, dice_train)
    mlp.fit(train_chunk, target)

    # Time to test the algorithm!
    test(mlp, dice_song, dice_member, dice_train, max_song_length, min_time, max_time, min_year, max_year)


def test(mlp, dice_song, dice_member, dice_train, max_song_length, min_time, max_time, min_year, max_year):
    # ['id', 'msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'target'] test
    test_path = '/home/lydia/PycharmProjects/untitled/currently using/test_with_target.h5'
    song_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs_test.h5'
    member_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members_test.h5'

    prev = 0
    i = 0
    start_time = time.time()
    while prev + 30000 < 2556790:
        song_chunk = manipulate_song(chunk_read(song_path, prev, prev + 30000), dice_song, max_song_length)
        member_chunk = manipulate_member(chunk_read(member_path, prev, prev + 30000), dice_member, min_time, max_time,
                                         min_year, max_year)
        test_chunk, target = manipulate_t(chunk_read(test_path, prev, prev + 30000), song_chunk,
                                               member_chunk, dice_train)
        test_chunk = test_chunk.drop(['id'], axis=1)
        target = target.reset_index(drop=True)
        nan_index = target[target.apply(np.isnan)]
        nan_index = nan_index.index.tolist()
        target = target.dropna()
        print(test_chunk[test_chunk.isna().any(axis=1)])
        prediction = mlp.predict(test_chunk)
        prediction = np.delete(prediction, nan_index)
        score = correct(prediction, target)
        prediction = pd.DataFrame(prediction)
        #file_save(prediction, 'Score_v7.h5')
        prev = prev + 30000
        total_time = time.time() - start_time
        print(score)
        loading_bar_test(i, total_time, score)
        i = i + 1
    song_chunk = manipulate_song(chunk_read(song_path, prev, 2556790), dice_song, max_song_length)
    member_chunk = manipulate_member(chunk_read(member_path, prev, 2556790), dice_member, min_time, max_time,
                                     min_year, max_year)
    test_chunk, target = manipulate_t(chunk_read(test_path, prev, 2556790), song_chunk,
                                      member_chunk, dice_train)
    test_chunk = test_chunk.drop(['id'], axis=1)
    target = target.reset_index(drop=True)
    nan_index = target[target.apply(np.isnan)]
    nan_index = nan_index.index.tolist()
    target = target.dropna()
    prediction = mlp.predict(test_chunk)
    prediction = np.delete(prediction, nan_index)
    score = correct(prediction, target)
    prediction = pd.DataFrame(prediction)
    #file_save(prediction, 'Score_v7.h5')
    loading_bar_test(i, total_time, score)


warnings.filterwarnings('ignore')
songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
members_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members.h5'
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
songs_dice_path = '/home/lydia/PycharmProjects/untitled/python files/dice_dist_songs_list_90.txt'
members_dice_path = '/home/lydia/PycharmProjects/untitled/python files/dice_dist_members_list_90.txt'
train_dice_path = '/home/lydia/PycharmProjects/untitled/python files/dice_dist_train_list_90.txt'


train(songs_path, members_path, train_path, songs_dice_path, members_dice_path, train_dice_path)
