from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
from spotipy_functions import get_song_metadata
import lyricsgenius
import re
import time

genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY", timeout=20, sleep_time=0.2, retries=3)

import requests
url_1950 = 'https://tsort.info/music/yr1950.htm'

def lyrics_cleaner(text):
    """
    Paramters
    text: str representing the lyrics to be cleaned
    ----------------
    Description
    This function will take in the lyrics and remove any unwanted characters, such as brackets, new lines, and numbers.
    It will also remove any unwanted characters, such as punctuation and special characters.
    ----------------
    Returns
    cleaned_text: str representing the cleaned lyrics
    """
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    cleaned_text = re.sub(r'[\r]', '', cleaned_text)
    cleaned_text = re.sub(r'[\n]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\d+\s+Contributors', '', cleaned_text)
    cleaned_text = re.sub(r'\d+\s+Contributor', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
    return cleaned_text

def web_scrape_lyrics(url, dataset):
    """
    Parameters

    url: str representing a music charts web-page from tsort.info
    dataset: str representing the filepath of a csv file to save the scraped data
    ----------------
    Description

    This function web-scrapes top 100 billboard songs from a given URL formatted in the format of tsort.info.
    Using lyricsgenius module and a Genius API access key, it searches for the song based on title and artist,
    then retrieves the raw lyrics. 

    This function will then use custom spotipy functions to get spotify metadata for a given song,
    provided the metadata is available. 

    This function will then write the data to a pre-existing csv file, appending the new data.
    
    If the function encounters encounters Genius API errors, it will attempt up to 5 consecutive
    times before stopping the scraping process. The loop will also break if there is an error with
    reading the url.
    ----------------
    Returns
    final_year_loop: str representing the final URL that was scraped.
    -----------------
    """
    genius = lyricsgenius.Genius("nxV0Q_srG3ZDwK4LxliW5EhA1T3wPu5KDCCsn58DmkxIjebbC_DJ1sulZ1T9NZVW", timeout=30, sleep_time=0.5, retries=3)
    with open(dataset, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Track Name', 'Artist Name', 'Genre', 'Year', 'Lyrics', 'Spotify Genres', 'Release Date', 'Popularity', 'Explicit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        timeout = time.time() + 60 * 55
        df = pd.read_csv(dataset)
        consecutive_errors = 0
        while url is not None and consecutive_errors < 5 and time.time() < timeout:
            response = requests.get(url, verify = False)
            if response.status_code == 200:
                html_content = response.text
            else:
                print(f"Failed to retrieve the page. Status code: {response.status_code}")
                final_year_loop = url
                break
            soup = BeautifulSoup(html_content, 'html.parser')
            table_soup = soup.find('table', class_='songlist')

            if not table_soup:
                print("Table with class 'songlist' not found.")
                final_year_loop = url
                break
            
            rows = table_soup.find_all('tr')

            for row in rows:
                columns = row.find_all('td')
                song_title = columns[2].text.strip()
                artist = columns[1].text.strip()

                if song_title in df['Track Name'].values:
                    print(f"Song '{song_title}' by '{artist}' already exists in the dataset. Skipping...")
                    continue

                try:
                    song = genius.search_song(song_title, artist)

                    if song:
                        consecutive_errors = 0
                        spotify_metadata = get_song_metadata(song_title, artist)

                        if spotify_metadata:
                            writer.writerow({
                                'Track Name': lyrics_cleaner(song.title),
                                'Artist Name': lyrics_cleaner(song.artist),
                                'Lyrics': lyrics_cleaner(song.lyrics),
                                'Spotify Genres': spotify_metadata['genres'],
                                'Release Date': spotify_metadata['release_date'],
                                'Popularity': spotify_metadata['popularity'],
                                'Explicit': spotify_metadata['explicit'],
                            })
                            print(f"Song found: {song} by {artist}")

                        else:
                            print(f"Spotify metadata for '{song}' by '{artist}' not found. Skipping...")
                    
                    else:
                        print(f"Song '{song_title}' by '{artist}' not found in Genius. Skipping...")
                
                except Exception as e:
                    print(f"Error occurred while fetching lyrics for {song_title} by {artist}: {e}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= 5:
                        print("Too many consecutive errors. Stopping the scraping process.")
                        final_year_loop = url
                        break
                    continue

            url_string = url
            match = re.search(r'yr(\d{4})', url_string)

            if match:
                year = match.group(1)
                new_year = str(int(year) + 1)
                url = url.replace(year, new_year)

            else:
                print("Year not found in the URL.")
                final_year_loop = url
                break

    return final_year_loop