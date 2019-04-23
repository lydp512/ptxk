########################################################################################################################
##                                            THIS FUNCTION WAS MADE                                                  ##
##       IN ORDER TO CALCULATE THE SIZE OF THE CHUNKSIZE WHILE READING THE TRAIN/TEST FILES IN THE MAIN SCRIPT        ##
##                READING THE ENTIRE TRAIN/TEST FILES IS POSSIBLE, AND THERE WERE NO MEMORY ISSUES                    ##
##       HOWEVER, WHILE MANIPULATING THOSE FILES, DELETING ARRAYS WHICH WERE THEIR BYPRODUCTS, WAS IMPOSSIBLE         ##
##               THE MEMORY OF THE BYPRODUCTS, WAS BOUND TO THEM, AND IT WOULD ACCUMULATE OVER TIME                   ##
##    HENCE, THE TWO PANDAS SERIES WHICH ARE RETURNED BY THIS SCRIPT, CALCULATE HOW MANY TIMES A SONG_ID APPEARS      ##
##    ALL ARRAYS (SONGS, TRAIN, TEST, TRAIN_VALUE_COUNTS_SONG AND TEST_VALUE_COUNTS_SONG) ARE SORTED ON SONG_ID       ##
########################################################################################################################
import pandas as pd
################################################ READING THE FILES #####################################################


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def save(array, location):
    array.to_hdf(location, key='df', mode='w')


######################################################### MAIN #########################################################


# Loads and reads the files
path_prefix = '/home/lydia/PycharmProjects/untitled/currently using/'

print('\n' + "train and test file reading starting...")
train = file_read(path_prefix+"train_with_members_sorted_by_song_id.h5")
test = file_read(path_prefix+"test_with_members_sorted_by_song_id.h5")

print(set(list(train.columns)) ^ set(list(test.columns)))
print(list(test.columns) - list(train.columns))















arertetry
songs = file_read(path_prefix + 'filtered_songs.h5')

# Only a single column is needed, so the others are dropped
songs = songs['song_id']


# Counts how many times each song_id gets repeated in the train, and test files
train_array = train['song_id'].value_counts(sort=False)
test_array = test['song_id'].value_counts(sort=False)


# Now, the times a song_id can't be seen in one of the files must be also factored in
zero_mentions = songs[(~songs.isin(train.song_id))]
zero_mentions = zero_mentions.reindex(index=zero_mentions)
zero_mentions = zero_mentions.fillna(0)
train_array = pd.concat([train_array, zero_mentions])
# Remember! The index is none other but the song_id!
train_array = train_array.sort_index()


# Same process, but for the test file
zero_mentions = songs[(~songs.isin(test.song_id))]
zero_mentions = zero_mentions.reindex(index=zero_mentions)
zero_mentions = zero_mentions.fillna(0)
test_array = pd.concat([test_array,zero_mentions])
test_array = test_array.sort_index()


# Saves and that's it!
save(train_array, path_prefix + 'train_value_counts_song.h5')
save(test_array, path_prefix + 'test_value_counts_song.h5')