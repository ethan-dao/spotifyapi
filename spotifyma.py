# Environment variables
from dotenv import load_dotenv
import os
load_dotenv()
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Import libraries
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import time
import ast
from requests.exceptions import ReadTimeout

# Authentication using SpotifyClientCredentials
auth = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(auth = auth, requests_timeout = 10, retries = 10)

# Let's use trackIDs to get the info and audio features of a track, and then put it into a dataframe
def getTrackFeatures(trackID):
    try:
        # Authentication key
        spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        # Using track() and audio_features() calls on Spotify API
        # 1. We are going to get the track info and audio features of a sample song and turn it into a dataframe
        trackInfo = spotify.track(trackID)
        trackFeatures = spotify.audio_features(trackID)
        # Order matters; use get track on Spotify for Developers
        name = trackInfo['name']
        trackID1 = trackFeatures[0]['id']
        artist = trackInfo['album']['artists'][0]['name']
        releaseDate = trackInfo['album']['release_date']
        duration = trackInfo['duration_ms']
        explicit = trackInfo['explicit']
        popularity = trackInfo['popularity']
        # audio_features() to get audio features of the track
        danceability = trackFeatures[0]['danceability']
        energy = trackFeatures[0]['energy']
        acousticness = trackFeatures[0]['acousticness']
        key = trackFeatures[0]['key']
        tempo = trackFeatures[0]['tempo']
        # Making the final track variable for our data
        track = [name, trackID1, artist, releaseDate, duration, explicit, popularity, danceability, energy, acousticness, key, tempo]
        # Outputs a list
        print(track)
        time.sleep(2)
        return(track)
    except Exception as e:
        print(f'Error fetching track features for trackID {trackID}: {e}')
        return None

# Let's try this with a playlist; first we want to create a function that gets all the track IDs from the tracks in a playlist
def getTrackIDs(playlistID, maxTracks = 50):
    # Authentication key
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    # Store the track IDs in a list
    trackIDs = []
    # Get important information (playlist, track count, limit)
    playlist = spotify.playlist(playlistID)
    trackCount = playlist['tracks']['total']
    # Limit amount of max tracks (so we don't run into too many API requests)
    maxTracks = min(maxTracks, trackCount)
    # For loop to get track IDs for each track in playlist
    for offset in range(0, maxTracks, 100):
        tracks = spotify.playlist_tracks(playlistID, offset=offset, limit = 100) # Process 100 tracks at a time to not exceed limit rates
        for items in tracks['items']:
            if items.get('track') and items['track'].get('id'):
                trackIDs.append(items['track']['id'])
                if len(trackIDs) > maxTracks:
                    break # Stop if we have reached maximum number of tracks
            if len(trackIDs) >= maxTracks:
                break # Stop if we have reached maximum number of tracks
    # Return final result
    return(trackIDs)

# Now, we can go through the track IDs to get the features of each track, looping through the whole playlist and putting it into a list
def getPlaylistFeatures(playlistID):
    # Authentication key
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(), requests_timeout = 10, retries = 10)
    # Define variables
    MAX_RETRIES = 5
    retryDelay = 5
    trackData = []
    for attempt in range(MAX_RETRIES):
        try:
            playlistTracks = spotify.playlist_tracks(playlistID)
            for track in playlistTracks['items']:
                trackName = track['track']['name']
                trackID = track['track']['id']
                trackArtists = [artist['name'] for artist in track['track']['artists']]
                trackAlbum = track['track']['album']['name']
                trackDuration = track['track']['duration_ms']
                trackPopularity = track['track']['popularity']
                # Getting audio features
                audioFeatures = spotify.audio_features(trackID)
                if audioFeatures:
                    audioFeatures = audioFeatures[0]
                    # Get genre for first artist on the track
                    artistID = track['track']['artists'][0]['id']
                    artistInfo = spotify.artist(artistID)
                    artistGenres = artistInfo.get('genres', []) # Create an empty list to store artist genres
                    # Add track and audio features to the list
                    trackFeatures = ({
                        'Track Name': trackName,
                        'Track ID': trackID,
                        'Artists': ', '.join(trackArtists),
                        'Album': trackAlbum,
                        'Genre': ', '.join(artistGenres),
                        'Duration (ms)': trackDuration,
                        'Popularity': trackPopularity,
                        'Danceability': audioFeatures['danceability'],
                        'Energy': audioFeatures['energy'],
                        'Acousticness': audioFeatures['acousticness'],
                        'Instrumentalness': audioFeatures['instrumentalness'],
                        'Liveness': audioFeatures['liveness'],
                        'Loudness': audioFeatures['loudness'],
                        'Speechiness': audioFeatures['speechiness'],
                        'Key': audioFeatures['key'],
                        'Tempo': audioFeatures['tempo']
                    })
                    trackData.append(trackFeatures)
                    print(trackFeatures)
                    time.sleep(1)
                else:
                    print(f'Audio features for track {trackID} not available')
            break # Break out of loop once playlist is processed
        except Exception as e:
            print(f'Error processing playlist {playlistID}: {e}')
            if 'rate limiting' in str(e).lower() and 'Retry-After' in getattr(e.response, 'headers', {}):
                retry_after_seconds = int(e.response.headers['Retry-After'])
                print(f'Rate limit exceeded. Retry after {retry_after_seconds} seconds...')
                time.sleep(retry_after_seconds)  # Wait for the specified duration
            elif attempt < MAX_RETRIES - 1:
                time.sleep(retryDelay)
                retryDelay *= 2
                print('Retrying...')
            else:
                print('Max retries reached. Skipping to next playlist')
                break
    return trackData

# Couldn't find any up-to-date datasets for songs to put in our recommendation system; we need to look at Spotify itself
# Generating a dataframe with all songs from Spotify's playlists; ensures that music is popular and up-to-date
def getSpotifyPlaylists():
    # Authentication key
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(), requests_timeout = 10, retries = 10)
    # Empty list for Spotify song IDs, set offset (since API limit is 50 playlists at a time)
    playlistIDList = []
    offset = 0
    # Get songs using Spotipy function
    while True:
        # While true, get playlists; if no more playlists, break out of while loop
        playlists = spotify.user_playlists('spotify', offset = offset)
        if not playlists['items']:
            break
        # Loop through playlists to get IDs
        for playlist in playlists['items']:
            playlistID = playlist['id']
            playlistIDList.append(playlistID)
            time.sleep(0.1)
        # Increment the offset by how many songs we looped through
        offset += len(playlists['items'])

    print(playlistIDList)
    return playlistIDList

def getSpotifySongs(playlistIDList):
    # Authentication key
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    # Variables
    MAX_RETRIES = 5
    retryDelay = 5
    trackDataList = []
    # Let's get all the playlist features from the playlist list
    for playlistID in playlistIDList:
        playlistFeatures = getPlaylistFeatures(playlistID)
        trackDataList.extend(playlistFeatures)
        print(f'Playlist {playlistID} done')
        time.sleep(30)
    return(trackDataList)

# Get Spotify playlists
playlistList = getSpotifyPlaylists()
# Initialize an empty DataFrame to store all songs
all_songs = pd.DataFrame()
# Define a batch size
batchSize = 15

# Loop through the playlist ranges
for start_offset in range(510, len(playlistList), batchSize):
    # Create a subset of playlistIDs for the current range
    subset_playlists = playlistList[start_offset:start_offset + batchSize]
    
    # Call the function for the current range of playlists
    subset_songs = getSpotifySongs(subset_playlists)
    subset_songs_df = pd.DataFrame(subset_songs)

    # Concatenate the results to the 'all_songs' DataFrame
    all_songs = pd.concat([all_songs, subset_songs_df])

    # Save temporary CSV
    if (start_offset + batchSize) % batchSize == 0 or (start_offset + batchSize) >= len(playlistList):
        csvFilename = f'spotifysampledata_{start_offset + batchSize}.csv'
        all_songs.to_csv(csvFilename, index = False)
        print(f'Saved CSV: {csvFilename}')

all_songs.to_csv('spotifysampledata_final.csv', index = False)








# Appending CSV files to each other
# dfCSVAppend = pd.DataFrame()
# csvList = ['spotifysampledata_0390.csv', 'spotifysampledata_405.csv', 'spotifysampledata_420.csv', 'spotifysampledata_435.csv', 
#            'spotifysampledata_450.csv', 'spotifysampledata_465.csv', 'spotifysampledata_480.csv', 'spotifysampledata_495.csv', 
#            'spotifysampledata_510.csv']
# # Append CSV's to each other
# for file in csvList:
#     df = pd.read_csv(file)
#     dfCSVAppend = pd.concat([dfCSVAppend, df], ignore_index=True)

# dfCSVAppend.to_csv('spotifysampledata_0510.csv', index=False)












