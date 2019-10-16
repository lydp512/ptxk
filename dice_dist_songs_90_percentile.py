import pandas as pd
import pickle


############################################### READ AND WRITE FUNCTIONS ###############################################
def file_read(location):
    df = pd.read_hdf(location, 'df')
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


################################################### SIMPLE FUNCTIONS ###################################################
# What popular row does, is to create a row, where the columns correspond to each genre/artist/language.
# Then, the most popular ones, are represented with an 1, while he others with a 0.
def popular_row(array, unique):
    # Creates a Series, full of zeroes.
    popular_values = pd.Series(data=0, index=unique)
    # The array index correspond to the top values, so those are marked with an 1.
    popular_values.loc[list(array.index.values)] = 1
    # Converts to dataframe, since series can't be transposed, then transposes and returns.
    popular_values = pd.DataFrame(popular_values)
    popular_values = popular_values.transpose()
    return popular_values


############################################## HEAVY LIFTING BEGINS HERE ###############################################
def count(array, column):
    # Very helpful, since there can be songs with values missing.
    # For example, a song without lyrics, will not have a lyricist!
    array = array.fillna('Unknown')
    # Multiple names of each row are expanded into lists
    # Language is an easy and fast column, so it's treated differently
    if column == 'language':
        # Simply counts the values, keeps a list of all the unique ones for the popular_row function, and saves the
        # top percentile as cn
        cn = array.value_counts()
        un = list(cn.index.values)
        cn = cn[cn > cn.quantile(.90)]
        cn = popular_row(cn, un)
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
        total = pd.DataFrame()
        # After the column is split to multiple columns, we count values for each and every one of them.
        # Then, we simply sum them up.
        for extra_column in array:
            add = array[extra_column].value_counts()
            if total.empty:
                total = add
            else:
                total = pd.concat([total, add], axis=1, sort=False)
            # Deletes just to be more memory sufficient
            del array[extra_column]
        # Genre_ids are just numbers, so doing all of this isn't necessary.
        if column != 'genre_ids':
            # "trims up" the names of the artists. Empty spaces are disregarded, everything is converted to uppercase
            # and artists that may have appeared twice in the data(e.g. john and JOHN) are summed up in a single column.
            total = total.fillna(0.0)
            total['names'] = total.index
            total['names'] = total['names'].apply(str)
            total['names'] = total['names'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
            total['names'] = total['names'].str.strip()
            total = total[total['names'].apply(lambda x: len(x) >= 2)]
            total['names'] = total['names'].str.upper()
            total = total.set_index('names')
            total = total.groupby(total.index).sum()
        else:
            # Simply changes Unknown to UNKNOWN
            total['names'] = total.index
            total['names'] = total['names'].str.upper()
            total = total.set_index('names')
        # Now that every row corresponds to an artist, it sums all the columns up, to a single column per artist.
        total = total.sum(axis=1)
        # Before we only keep the top 10 quantile, all the unique values are saved to a list
        all_un_names = list(total.index.values)
        # Drops the bottom 90%
        total = total[total > total.quantile(.90)]
        # Creates the most common row!
        cn = popular_row(total, all_un_names)
    return cn


def find_percentile(df, test_df):
    df = pd.concat([df, test_df], axis=0, ignore_index=True)
    # Not every column in the songs file is needed to calculate dice distance.
    # For example, values such as song_length exist, which is much easier to normalize
    relevant_columns = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    dice = []
    # For every column, it saves the top10 quantile values with an 1, while the rest are saved with a 0
    for column in relevant_columns:
        dice.append(count(df[column], column))
    return dice


################################################### MAIN STARTS HERE ###################################################
songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
songs_test_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs_test.h5'
songs = file_read(songs_path)
songs_test = file_read(songs_test_path)
# Does magic! (not really)
list_of_dice = find_percentile(songs, songs_test)
# Saves and done!
save_list(list_of_dice, 'dice_dist_list_90.txt')
