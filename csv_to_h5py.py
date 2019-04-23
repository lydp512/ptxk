import pandas as pd
import h5py
import csv



file_list = ['test', 'train', 'members', 'sample_submission', 'songs']
file_list_csv = [s + '.csv' for s in file_list]
file_list_h5py = [s + '.h5' for s in file_list]
for i in range(len(file_list)):
    if (file_list_csv[i] == 'songs.csv'):
        df = pd.read_csv(file_list_csv[i], names=['song_id', 'song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist',
                                          'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE)
    else:
        # joins string into full file path
        df = pd.read_csv(file_list_csv[i], delimiter=',')
    df.to_hdf(file_list_h5py[i], key='df', mode='w')
    del df  # allow df to be garbage collected
    print("ok")