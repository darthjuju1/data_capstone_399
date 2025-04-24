import lyricsgenius
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from dotenv import load_dotenv # Load the environment file .env in order to get enviornment variables (in this case client ID and secret)
import os # Necessary for loading an enviornment
import base64 # Necessary for encoding the client ID and secret
import requests # Necessary for making a POST request to the Spotify API and web-scraping
import json # Necessary for parsing the JSON response from the Spotify API
import time


genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY")

def generate_csv(top_songs, file_path, start_index=0, create_copy=False, artist_name = 'Artist Name(s)', track_name = 'Track Name'):
    """
    Parameters 
    top_songs: CSV containing the top songs
    file_path: Name of the file to write to
    start_index: Index to start from in the DataFrame (default=0)
    create_copy: If True, creates a copy of the CSV instead of overwriting (default=False)
    artist_name: Column name for artist names (default='Artist Name(s)')
    track_name: Column name for track names (default='Track Name')
    ----------------
    Description 
    This function will take in a CSV containing the top songs and write the song titles and artist names to a CSV file.
    Needs LyricsGenius to be installed, as well as a Genius API key and a specifically formatted CSV file.
    ----------------
    Returns
    If create_copy=False:
        index = The index of the last song written to the CSV file
    If create_copy=True:
        (index, copy_path) = The index of the last song written and the path to the copied CSV file
    """
    from spotipy_functions import get_song_metadata
    import os
    
    df = pd.read_csv(file_path)
    existing_songs = set(zip(df[track_name].str.lower(), df[artist_name].str.lower()))
    if create_copy:
        file_name, file_ext = os.path.splitext(file_path)
        copy_path = f"{file_name}_copy{file_ext}"
        output_path = copy_path
    else:
        output_path = file_path
    
    artist_title = top_songs[artist_name]
    track_title = top_songs[track_name]
    genres = top_songs['Artist Genres']
    years = top_songs['Album Release Date']

    count = start_index
    error_count = 0
    lyrics_list = []
    timeout = time.time() + 60 * 55

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Track Name', 'Artist Name', 'Genre', 'Year', 'Lyrics', 'Spotify Genres', 'Release Date', 'Popularity', 'Explicit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        while time.time() < timeout and count < len(track_title):
            song_title = track_title[count]
            artist = artist_title[count]
            year = years[count]
            
            try:
                if song_title in existing_songs:
                    count += 1
                    error_count += 1
                    print(f"Song '{song_title}' by '{artist}' already exists in the dataset. Skipping...")
                    continue
                else:
               
                    song = genius.search_song(song_title, artist)
                    if song:
                        error_count = 0
                      
                        spotify_metadata = get_song_metadata(song_title, artist)
                        if spotify_metadata:
                            writer.writerow({
                                'Track Name': song.title,
                                'Artist Name': song.artist,
                                'Genre': genres[count], 
                                'Year': year, 
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
                        error_count += 1
                        if error_count > 5:
                            print("Too many errors. Stopping the process.")
                            break
                        count += 1
            except Exception as e:
                print(f"Error occurred while fetching lyrics for {song_title} by {artist}: {e}")
                count += 1
        print(f"Collected lyrics for {len(lyrics_list)} songs.")
    
    if create_copy:
        return count, copy_path
    else:
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

def csv_merger(csv1, csv2):
    """
    Parameters
    csv1: The first CSV file to merge
    csv2: The second CSV file to merge
    ----------------
    Description
    This function merges two CSV files into one and deletes the second file.
    The merged file will be saved as the first file.
    ----------------
    Returns
    merged_csv: The merged DataFrame
    """
    merged_csv = pd.concat([pd.read_csv(csv1), pd.read_csv(csv2)], ignore_index=True)
    merged_csv.to_csv(csv1, index=False)
    
    if os.path.exists(csv2):
        os.remove(csv2)
        print(f"{csv2} has been deleted.")
    else:
        print(f"{csv2} does not exist.")
    
    return merged_csv

def count_rows_in_csv(file_path):
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    return row_count