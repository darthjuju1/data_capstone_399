import lyricsgenius
import requests
import pandas as pd
from spotipy_functions import get_song_metadata
import csv
from bs4 import BeautifulSoup
import os
import re

"""
dataset = 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/song_lyrics - Copy.csv'
url = 'https://tsort.info/music/yr1950.htm'
genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY", timeout=20, sleep_time=0.2, retries=3)
temp_count = 0
while temp_count <10:
    temp_count += 1
    with open(dataset, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Track Name', 'Artist Name', 'Genre', 'Year', 'Lyrics', 'Spotify Genres', 'Release Date', 'Popularity', 'Explicit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        df = pd.read_csv(dataset)
        response = requests.get(url, verify = False)
        if response.status_code == 200:
            html_content = response.text
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
        soup = BeautifulSoup(html_content, 'html.parser')
        table_soup = soup.find('table', class_='songlist')

        if table_soup:
            rows = table_soup.find_all('tr')

            for row in rows:
                columns = row.find_all('td')
                song_title = columns[2].text.strip()
                artist = columns[1].text.strip()
                if song_title in df['Track Name'].values:
                    print(f"Song '{song_title}' by '{artist}' already exists in the dataset. Skipping...")
                    continue
                else:
                    try:
                        song = genius.search_song(song_title, artist)
                        
                        if song:
                            spotify_metadata = get_song_metadata(song_title, artist)
                            if spotify_metadata:
                                writer.writerow({
                                    'Track Name': song.title,
                                    'Artist Name': song.artist,
                                    'Lyrics': song.lyrics,
                                    'Spotify Genres': spotify_metadata['genres'],
                                    'Release Date': spotify_metadata['release_date'],
                                    'Popularity': spotify_metadata['popularity'],
                                    'Explicit': spotify_metadata['explicit'],
                                })
                                print(f"Song found: {song} by {artist}")
                            else:
                                print(f"Spotify metadata for '{song}' by '{artist}' not found. Skipping...")
                        else:
                            print(f"Song '{song}' by '{artist}' not found in Genius. Skipping...")
                    except Exception as e:
                        print(f"Error occurred while fetching lyrics for {song} by {artist}: {e}")
                        continue
        else:
            print("Table with class 'songlist' not found.")

"""
url = 'https://tsort.info/music/yr1950.htm'

