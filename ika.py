import pandas as pd
import warnings


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


def make_dummies(array, dict_for_map, names, column_name):
    array = array.str.split(pat="|", expand=True)
    for column in array:
        array[column] = array[column].map(dict_for_map)
    array = array.astype('category', categories = names)
    array = pd.get_dummies(array.apply(pd.Series).stack(), prefix=column_name).sum(level=0)
    return array


def manipulate_song(array, maping, listing, language_list, max_length):
    song_length = array['song_length'].astype(float)
    song_length = song_length.fillna(0)
    # min length is 0
    song_length = song_length.div(max_length)
    languages = array['language']
    languages = languages.astype('category', categories=language_list)
    languages = pd.get_dummies(languages)
    entire_file = array.drop(['song_length', 'language'], axis=1)
    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    entire_file = pd.concat([entire_file, song_length], axis=1)
    i = 0
    for column in column_list:
        column = make_dummies(entire_file[column], maping[i], listing[i], column)
        entire_file = pd.concat([entire_file, column], axis=1)
        i = i + 1
    entire_file = pd.concat([entire_file, languages], axis=1)
    entire_file = entire_file.drop(['genre_ids', 'artist_name', 'composer', 'lyricist'], axis=1)
    return entire_file


def manipulate_train(array, categories):
    relevant_categories = ['source_system_tab', 'source_screen_name', 'source_type']
    for i in range(3):
        column = array[relevant_categories[i]].astype('category', categories=categories[i])
        column = pd.get_dummies(column)
        array = pd.concat([array,column], axis=1)
    array = array.drop(relevant_categories, axis=1)
    return array


def train_them(train, songs, maping, listing, language_list, max_length):
    prev = 0
    while prev + 1000 < 7377416:
        new_song = manipulate_song(chunk_read(songs, prev, prev+1000), maping, listing, language_list, max_length)
        new_train, target = manipulate_train(chunk_read(train, prev, prev+1000))
        new_train = train_with_songs(new_train, new_song)



############################################## MAIN STARTS HERE ################################################
warnings.filterwarnings('ignore')

prev=1
train = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
trains = file_read(train)
categories = []
for column in ['source_system_tab', 'source_screen_name', 'source_type']:
    trains[column] = trains[column].fillna('Unknown')
    categories.append(trains[column].unique())
del trains
new_train = manipulate_train(chunk_read(train, prev, prev+1000), categories)

songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
songs = file_read(songs_path)
songs = songs[1:]
un_languages = songs['language'].unique()
song_length = pd.to_numeric(songs['song_length'])
max_song_length = song_length.max()
max_song_length = float(max_song_length)
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
percentile_map, percentile_list = find_percentile(songs)
del songs

train_them(train_path, songs_path, percentile_map, percentile_list, un_languages, max_song_length)
