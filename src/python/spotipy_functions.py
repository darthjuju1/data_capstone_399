import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import pandas as pd
from dotenv import load_dotenv
import os
import base64
import requests

load_dotenv()
client_id = os.getenv("client_id") #referencing client id and secret from the .env file in quotes for some reason
client_secret = os.getenv("client_secret")

def get_token():
    """
    Parameters 
    None
    ----------------
    Description 
    This function encodes the client_id and client_secret and makes a POST request to the Spotify API to get an access token.
    ----------------
    Returns
    token: Spotify API access token
    """
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    base_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + auth_base64,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    response = requests.post(base_url, headers=headers, data=data)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.content}")
        return None
    
    json_result = response.json() 

    if 'access_token' in json_result:
        token = json_result['access_token']
        return token
    else:
        print("Error: No access_token found in response")
        print(f"Response content: {json_result}")
        return None

def get_auth_header(token):
    return {'Authorization': f'Bearer {token}'}

token = get_token()
sp = spotipy.Spotify(auth=token)

def get_song_id(song_title, artist_name):
    """
    Parameters
    song_title: str
    artist_name: str
    ----------------
    Description
    This function takes in a song title and artist name and returns the Spotify ID for the song.
    ----------------
    Returns
    song_id: str
    """
    query = f'track:{song_title} artist:{artist_name}'
    results = sp.search(q=query, limit=1, type = 'track')
    if results['tracks']['items']:
        song_id = results['tracks']['items'][0]['id']
        return song_id
    else:
        return None

song_id = get_song_id("Blinding Lights", "The Weeknd")
if song_id:
    print(f"Song ID: {song_id}")
else:
    print("Song not found.")

def get_song_metadata(song_title, artist_name):
    """
    Fetch song metadata such as genre, artist, release date, popularity, and explicit status.
    """
    song_id = get_song_id(song_title, artist_name)
    
    if song_id:
        # Fetch track info from Spotify
        track = sp.track(song_id)
        
        # Genre: Try to fetch from album, otherwise use artist genres
        album_genres = track['album'].get('genres', [])
        if album_genres:
            genre = album_genres
        else:
            artist_data = sp.artist(track['artists'][0]['id'])
            genre = artist_data.get('genres', ['N/A'])
        
        # Release Date
        release_date = track['album']['release_date']
        
        # Popularity
        popularity = track['popularity']
        
        # Explicit
        explicit = track['explicit']
        
        # Artist(s)
        artists = ', '.join([artist['name'] for artist in track['artists']])
        
        # Return metadata as a dictionary
        metadata = {
            'track': song_title,
            'lyrics': 'Lyrics placeholder',  # You can replace with your lyrics field here
            'genres': ', '.join(genre),  # Join genres in case multiple are available
            'artist': artists,
            'release_date': release_date,
            'popularity': popularity,
            'explicit': explicit
        }
        
        return metadata
    else:
        print(f"Song '{song_title}' by {artist_name} not found.")
        return None