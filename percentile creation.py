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
    
    


    
train_path = '/home/lydia/PycharmProjects/untitled/old uses/train.h5'
trains = file_read(train_path)
categories = []
for column in ['source_system_tab', 'source_screen_name', 'source_type']:
    trains[column] = trains[column].fillna('Unknown')
    categories.append(trains[column].unique())
del trains
save_list(categories, 'categories.txt')


songs_path = '/home/lydia/PycharmProjects/untitled/currently using/repeated_songs.h5'
songs = file_read(songs_path)
songs = songs[1:]
un_languages = songs['language'].unique()
song_length = pd.to_numeric(songs['song_length'])
max_song_length = song_length.max()
max_song_length = float(max_song_length)
save_list(max_song_length, 'max_song_length.txt')
percentile_map, percentile_list = find_percentile(songs)
percentile_list.append(un_languages)
save_list(percentile_list, 'percentile_list.txt')
save_list(percentile_map, 'percentile_map.txt')
del songs
