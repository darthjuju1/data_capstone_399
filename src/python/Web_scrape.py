from bs4 import BeautifulSoup
import requests
import pandas as pd


import requests
url = 'https://tsort.info/music/yr1951.htm'

# Send an HTTP request to the URL
response = requests.get(url, verify=False)

if response.status_code == 200:
    html_content = response.text
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

soup = BeautifulSoup(html_content, 'html.parser')

table_soup = soup.find('table', class_='songlist')

if table_soup:
    # Find all rows in the table
    rows = table_soup.find_all('tr')

    # Loop through each row and extract song and artist names
    for row in rows:
        columns = row.find_all('td')  # Find all columns in the row
        if len(columns) >= 2:  # Ensure there are at least two columns (song and artist)
            song = columns[2].text.strip()  # First column is the song name
            artist = columns[1].text.strip()  # Second column is the artist name
            print(f"Song: {song}, Artist: {artist}")
else:
    print("Table with class 'songlist' not found.")

#print(table_soup.prettify())
#print(table_soup.prettify())
