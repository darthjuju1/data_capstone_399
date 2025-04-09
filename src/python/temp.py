from genius_functions import generate_csv, count_rows_in_csv, csv_merger
from Web_scrape import web_scrape_lyrics
import lyricsgenius

csv_merger('C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics.csv', 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics_copy.csv')

#web_scrape_lyrics('https://tsort.info/music/yr1950.htm', 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics - Copy.csv')

#song = genius.search_song("Tennessee Waltz", "Patti Page")
#print(song.lyrics)