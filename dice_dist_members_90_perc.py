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
def count(array):
    # Very helpful, since there can be songs with values missing.
    array = array.fillna('Unknown')
    # Simply counts the values, keeps a list of all the unique ones for the popular_row function, and saves the
    # top percentile as cn
    cn = array.value_counts()
    un = list(cn.index.values)
    cn = cn[cn > cn.quantile(.90)]
    cn = popular_row(cn, un)
    return cn


def find_percentile(df):
    # Only two values require dice distance now
    relevant_columns = ['city', 'registered_via']
    dice = []
    # For every column, it saves the top10 quantile values with an 1, while the rest are saved with a 0
    for column in relevant_columns:
        dice.append(count(df[column]))
    return dice


################################################### MAIN STARTS HERE ###################################################
members_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_members.h5'
members = file_read(members_path)
# Does magic! (not really)
list_of_dice = find_percentile(members)
# Saves and done!
save_list(list_of_dice, 'dice_dist_members_list_90.txt')
