import pandas as pd
import numpy as np
import warnings
import pickle

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


################################################# REPEATING FUNCTIONS ##################################################
def shuffled(df):
    # drops NaN values on target (can't train with non data)
    df = df[df['target'].notna()]
    if 'id' in df.columns:
        # drops id (it's the same value as the index)
        df = df.drop(['id'], axis=1)
    # shuffles test
    df = df.sample(frac=1)
    return df


def repeat_members(tes, mem):
    # saves original shuffled index
    indexes = tes.index

    # sorts values based on msno (alphabetically)
    tes = tes.sort_values(by=['msno'])

    # this is how many times an id repeats
    counts = pd.DataFrame(tes['msno'].value_counts()).reset_index().sort_values(by=['index'])

    # gets unique ids existing in tes (there are more values in 'mem', which are not needed now)
    unique_msno = tes['msno'].unique()
    # keeps only the needed rows, and sorts
    mem = mem[mem['msno'].isin(unique_msno)]
    mem = mem.sort_values(by=['msno'])

    # re indexes to match tes
    mem = mem.reindex(mem.index.repeat(counts['msno']))
    mem = mem.set_index(tes.index)

    # re indexes mem according to the original test shuffle
    mem = mem.reindex(np.array(indexes))
    tes = tes.reindex(np.array(indexes))
    print(mem['msno'], tes['msno'])
    file_save(mem, 'member_proper_order_jul.h5')


def shave(mem, son, tes):
    # now the other way around
    unique = tes['msno'].unique()
    mem = mem[mem['msno'].isin(unique)]

    # one final time
    unique = tes['song_id'].unique()
    son = son[son['song_id'].isin(unique)]

    # gets unique ids existing in the template (there are more unique values in the repeated arrays,
    # which are not needed now)
    unique = mem['msno'].unique()
    # keeps only the needed rows, and sorts
    tes = tes[tes['msno'].isin(unique)]

    # does the same with son
    unique = son['song_id'].unique()
    tes = tes[tes['song_id'].isin(unique)]

    return mem, son, tes


def repeat(template, to_be_repeated, id, filename):
    # saves original shuffled index
    indexes = template.index

    # sorts values based on id (alphabetically)
    template = template.sort_values(by=[id])

    # this is how many times an existing id repeats
    unique = to_be_repeated[id].unique()
    template = template[template[id].isin(unique)]
    counts = pd.DataFrame(template[id].value_counts()).reset_index().sort_values(by=['index'])

    # gets unique ids existing in the template (there are more unique values in the repeated arrays,
    # which are not needed now)
    unique = template[id].unique()
    # keeps only the needed rows, and sorts
    to_be_repeated = to_be_repeated[to_be_repeated[id].isin(unique)]
    to_be_repeated = to_be_repeated.sort_values(by=[id])

    # re indexes to match template
    to_be_repeated = to_be_repeated.reindex(to_be_repeated.index.repeat(counts[id]))
    to_be_repeated = to_be_repeated.set_index(template.index)

    # re indexes repeated array according to the original test shuffle
    to_be_repeated = to_be_repeated.reindex(indexes)
    template = template.reindex(np.array(indexes))
    file_save(to_be_repeated, filename)
    if id == 'msno':
        file_save(template, 'train_proper_order_jul.h5')
    else:
        file_save(template, 'test_proper_order_jul.h5')
    del to_be_repeated


######################################################## MAIN #########################################################
warnings.filterwarnings('ignore')


# reads the files
train_path = '/home/lydia/PycharmProjects/untitled/currently using/train.h5'
# [id, msno, song_id, source_system_tab, source_screen_name, source_type, target]
song_path = '/home/lydia/PycharmProjects/untitled/currently using/songs.h5'
# [id, song_id, song_length, genre_ids, artist_name, composer, lyricist, language]
member_path = '/home/lydia/PycharmProjects/untitled/currently using/members.h5'
# [msno, city, bd, gender, registered_via, registration_init_time, expiration_date]

# reads the files (song file has an extra line that needs to be dropped)
member = file_read(member_path)
song = file_read(song_path).iloc[1:]
train = file_read(train_path)

# checks and deletes rows
member, song, train = shave(member, song, train)

# shuffles (must randomize input)
train = shuffled(train)

repeat(train, member, 'msno', 'train_member_proper_order_jul.h5')
repeat(train, song, 'song_id', 'train_song_proper_order_jul.h5')



# now same thing for the test file
test_path = '/home/lydia/PycharmProjects/untitled/currently using/test_with_target.h5'
# [id, msno, song_id, source_system_tab, source_screen_name, source_type, target]
song_path = '/home/lydia/PycharmProjects/untitled/currently using/songs.h5'
# [id, song_id, song_length, genre_ids, artist_name, composer, lyricist, language]
member_path = '/home/lydia/PycharmProjects/untitled/currently using/members.h5'
# [msno, city, bd, gender, registered_via, registration_init_time, expiration_date]

# reads the files (song file has an extra line that needs to be dropped)
member = file_read(member_path)
song = file_read(song_path).iloc[1:]
test = file_read(test_path)

# checks and deletes rows
member, song, test = shave(member, song, test)

# shuffles (must randomize input)
test = shuffled(test)

repeat(test, member, 'msno', 'test_member_proper_order_jul.h5')
repeat(test, song, 'song_id', 'test_song_proper_order_jul.h5')

