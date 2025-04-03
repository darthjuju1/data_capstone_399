from genius_functions import generate_csv, count_rows_in_csv, csv_merger
from spotipy_functions import mini_metadata_acquirer, get_song_id, get_audio_features, get_song_metadata

#csv_merger('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv', 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics_copy.csv')

mini_metadata_acquirer('Det Regner i Oslo', 'Astrid S')
