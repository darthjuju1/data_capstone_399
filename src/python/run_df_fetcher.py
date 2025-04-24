import lyricsgenius
import pandas as pd

# Edit this path location to a given csv file.
given_df = pd.read_csv('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/top_10000_1950-now.csv')

genius = lyricsgenius.Genius("nxV0Q_srG3ZDwK4LxliW5EhA1T3wPu5KDCCsn58DmkxIjebbC_DJ1sulZ1T9NZVW", timeout=20, sleep_time=0.2, retries=3)

from genius_functions import generate_csv, count_rows_in_csv, csv_merger

row_count = count_rows_in_csv('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv')
generate_csv(given_df, file_path = 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv',start_index=row_count)

