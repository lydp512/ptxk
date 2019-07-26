from sklearn.neural_network import MLPClassifier
import pandas as pd
import warnings
import pickle


############################################ SIMPLE READING/SAVING FUNCTIONS ###########################################
def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def chunk_read(location, start, stop):
    df = pd.read_hdf(location, 'df', start=start, stop=stop)
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
    array = pd.get_dummies(array.apply(pd.Series).stack(), prefix=column_name).sum(level=0)
    # Creates more columns for dummy variables that are not available
    array = array.T.reindex(names).T.fillna(0)
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
    array['language'] = array['language'].T.reindex(listing[4]).T.fillna(0)
    column_list = ['genre_ids', 'artist_name', 'composer', 'lyricist']
    i = 0
    # Makes dummies for each "more demanding" column
    for column in column_list:
        column = make_dummies(array[column], maping[i], listing[i], column)
        array = pd.concat([array, column], axis=1)
        i = i + 1
    # Drops the original columns, since now they're replaced with the dummy columns
    array = array.drop(['genre_ids', 'artist_name', 'composer', 'lyricist'], axis=1)
    return array


###################################### PREPARES THE TRAIN ARRAY FOR THE ALGORITHM ######################################
def manipulate_train(array, categories):
    # Separates the target from the array
    target = array['target']
    array = array.drop(['target'],axis=1)
    # These are the categories that need to get dummies
    relevant_categories = ['source_system_tab', 'source_screen_name', 'source_type']
    # Make the dummies happen
    for i in range(3):
        column = array[relevant_categories[i]]
        column = pd.get_dummies(column, prefix=relevant_categories[i])
        column = column.T.reindex(categories[i]).T.fillna(0)
        array = pd.concat([array,column], axis=1)
    # Drop the original columns, like before
    array = array.drop(relevant_categories, axis=1)
    return array, target


###################################################### MERGES THEM #####################################################
def train_with_songs(train, song):
    # Rather simple. Gets rid of the ids, and then merges them together. Simple as.
    song = song.drop(['song_id'],axis=1)
    train = train.drop(['msno', 'song_id'],axis=1)
    train = pd.concat([train,song],axis=1)
    train = train.fillna(0)
    return train


##################################################### TRAINS THEM #####################################################
def train_them(train, songs, maping, listing, max_length, categories):
    prev = 0
    mlp = MLPClassifier(warm_start=True)
    # Reads in 10k chunks
    while prev + 10000 < 7377418:
        # Takes care of the arrays first. Must be encoded, normalized and ready to go
        new_song = manipulate_song(chunk_read(songs, prev, prev+10000), maping, listing, max_length)
        new_train, target = manipulate_train(chunk_read(train, prev, prev+10000),categories)
        new_train = train_with_songs(new_train, new_song)
        # Feeds them into the algorithm, it's good to go!
        mlp.fit(new_train, target)
        prev = prev + 10000
    # Runs one last time. Same as before, it's just a smaller chunk now.
    new_song = manipulate_song(chunk_read(songs, prev, 7377418), maping, listing, max_length)
    new_train, target = manipulate_train(chunk_read(train, prev, 7377418), categories)
    new_train = train_with_songs(new_train, new_song)
    mlp.fit(new_train, target)
    # Done with the fitting! Time to put it to the test!



############################################## MAIN STARTS HERE ################################################
warnings.filterwarnings('ignore')

# Simply reads the files
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
file_path = '/home/lydia/PycharmProjects/untitled/currently using/'
songs_path = file_path + 'repeated_songs.h5'
percentile_map = read_list(file_path + 'percentile_map.txt')
categories = read_list(file_path + 'categories.txt')
max_song_length = read_list(file_path + 'max_song_length.txt')
percentile_list = read_list(file_path + 'percentile_list.txt')

# All read and good. Ready to go!
train_them(train_path, songs_path, percentile_map, percentile_list, max_song_length, categories)
