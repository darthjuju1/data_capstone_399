import lyricsgenius
import pandas as pd
import csv
import re
from Web_scrape import web_scrape_lyrics

#In time we will want to rename this to __main__.py I believe

top_10000 = pd.read_csv('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/top_10000_1950-now.csv')
genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY", timeout=20, sleep_time=0.2, retries=3)

from genius_functions import generate_csv, count_rows_in_csv, csv_merger

row_count = count_rows_in_csv('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv')
print(row_count)
generate_csv(top_10000, 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv',start_index=row_count, create_copy=True)
#csv_merger('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv', 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics - Copy.csv')


#web_scrape_lyrics('https://tsort.info/music/yr1950.htm', 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics - Copy.csv')

