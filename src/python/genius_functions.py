import lyricsgenius
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from dotenv import load_dotenv # Load the environment file .env in order to get enviornment variables (in this case client ID and secret)
import os # Necessary for loading an enviornment
import base64 # Necessary for encoding the client ID and secret
import requests # Necessary for making a POST request to the Spotify API  
import json # Necessary for parsing the JSON response from the Spotify API


genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY")


def generate_csv(top_songs, file_path, start_index=0):
    """
    Parameters 
    top_songs: DataFrame containing the top songs
    file_path: Name of the file to write to
    ----------------
    Description 
    This function will take in the DataFrame containing the top songs and write the song titles and artist names to a CSV file.
    Needs LyricsGenius to be installed, as well as a Genius API key and a specifically formatted CSV file.
    ----------------
    Returns
    file_name = CSV file with the song titles and artist names
    index = The index of the last song written to the CSV file
    """
    from spotipy_functions import get_song_metadata
    artist_name = top_songs['Artist Name(s)']
    track_title = top_songs['Track Name']
    genres = top_songs['Artist Genres']
    years = top_songs['Album Release Date']

    count = start_index
    lyrics_list = []

    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Track Name', 'Artist Name', 'Genre', 'Year', 'Lyrics', 'Spotify Genres', 'Release Date', 'Popularity', 'Explicit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        while len(lyrics_list) < 10 and count < len(track_title):
            song_title = track_title[count]
            artist = artist_name[count]
            year = years[count]
        
            if not song_title or not artist or isinstance(song_title, float) or isinstance(artist, float):
                print(f"Invalid data for song {count + 1}: '{song_title}' by '{artist}', skipping...")
                count += 1
                continue
            
            try:
                # Fetch lyrics from Genius
                song = genius.search_song(song_title, artist)
                
                if song:
                    # Get Spotify metadata
                    spotify_metadata = get_song_metadata(song_title, artist)
                    if spotify_metadata:
                        writer.writerow({
                            'Track Name': song.title,
                            'Artist Name': song.artist,
                            'Genre': genres[count],  # From the DataFrame
                            'Year': year,  # From the DataFrame
                            'Lyrics': song.lyrics,
                            'Spotify Genres': spotify_metadata['genres'],
                            'Release Date': spotify_metadata['release_date'],
                            'Popularity': spotify_metadata['popularity'],
                            'Explicit': spotify_metadata['explicit'],
                        })

                        lyrics_list.append(song.lyrics)
                        print(f"Song {count + 1} found: {song_title} by {artist}")
                        count += 1
                    else:
                        print(f"Spotify metadata for '{song_title}' by '{artist}' not found. Skipping...")
                        count += 1
                else:
                    print(f"Song '{song_title}' by '{artist}' not found in Genius. Skipping...")
                    count += 1
            except Exception as e:
                print(f"Error occurred while fetching lyrics for {song_title} by {artist}: {e}")
                count += 1
        print(f"Collected lyrics for {len(lyrics_list)} songs.")
    
    return count



def search_genius(search_term):
    """
    Parameters 
    access_token: Genius API access token
    search_term: term to search for
    ----------------
    Description 
    This function will take in the access token and search term and will search for the term in the Genius API.
    It will then print out the total number of results and the total number of pages.
    ----------------
    Returns
    None
    """
    base_url = 'https://api.genius.com/search'
    params = {
        'q': search_term,
        'per_page': 50,
        'page': 1
    }
    genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY", timeout=20, sleep_time=0.2, retries=3)
    headers = {
        'Authorization': f'Bearer {genius}'
    }
    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        total_results = data['response']['meta']['total_count']
        total_pages = (total_results // 50) + (1 if total_results % 50 > 0 else 0)
        print(f'Total Results: {total_results}')
        print(f'Total Pages: {total_pages}')

        for hit in data['response']['hits']:
            song_title = hit['result']['title']
            artist_name = hit['result']['primary_artist']['name']
            print(f'{song_title} by {artist_name}')
    else:
        print(f"Error: {response.status_code}")

    return 

def count_rows_in_csv(file_path):
    if not os.path.exists(file_path):
        return 0  # If the file doesn't exist, return 0
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader) # Subtract 1 to exclude the header
    return row_count