import csv
import lyricsgenius
from dotenv import load_dotenv
import spotipy
from spotipy_functions import get_song_id, get_token

token = get_token()
sp = spotipy.Spotify(auth=token)

genius = lyricsgenius.Genius("_F0e3onUyr31nflC3l2HUMOXl1xqX5ABSIPAd0PMpwLfy8ViEhxeDJESGV52IvlY")

def info_getter(list_of_searches):
    """
    Parameters 
    list_of_searches: list of tuples containing the song and artist to search for
    ----------------
    Description 
    This function will take in the search term and artist and will search for the song and artist in the Genius API.
    It will then write the artist, song, and lyrics to a csv file.
    ----------------
    Returns
    CSV file with the artist, song, and lyrics
    """
    with open("C:\\Users\\Daniel Day\\Downloads\\School\\Data-399\\data_capstone_399\\data\\lyrics.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Artist", "Song", "Lyrics"])

        for song_title, artist_name in list_of_searches:
            song = genius.search_song(song_title, artist_name)

            if song:
                writer.writerow([song.artist, song.title, song.lyrics])
                
            else:
                # If song is not found, write a message saying song was not found
                writer.writerow([artist_name, song_title, "Song not found"])
    return csvfile

def spotipy_testing(top_songs, file_path, start_index=0):
    count = start_index
    temp_count = 0
    song_names = top_songs['Track Name']
    artist_names = top_songs['Artist Name(s)']

    # Prepare CSV headers
    headers = ['Track Name', 'Artist Name', 'Song ID', 'Genre', 'Release Date']

    # Open the CSV file and write headers if the file is new or empty
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty, to write headers
            writer.writerow(headers)

    # Iterate through the song data
    while temp_count < 10:
        for song, artist in zip(song_names, artist_names):
            song_id = get_song_id(song, artist)
            if song_id:
                # Fetch track info from Spotify
                track = sp.track(song_id)
                album_genres = track['album'].get('genres', [])
                if album_genres:
                    genre = album_genres
                else:
                    # If no genres in the album, fall back to artist genres
                    artist_data = sp.artist(track['artists'][0]['id'])
                    genre = artist_data.get('genres', ['N/A'])
                release_date = track['album']['release_date']
                
                # Prepare the data row
                row = [song, artist, song_id, genre, release_date]
                writer.writerow(row)
                
                # Increment count for next iteration
                count += 1
            else:
                # Handle the case where no song ID is found
                print(f"Song '{song}' by {artist} not found.")
            
            # Optional: print progress
            if count % 10 == 0:
                print(f"Processed {count} songs...")

    print(f"Finished processing songs. Data saved to {file_path}.")