# Import libraries
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# Environment variables
from dotenv import load_dotenv
import os
load_dotenv()
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
# Authentication using SpotifyClientCredentials
auth = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(auth = auth, requests_timeout = 10, retries = 10)

# RECOMMENDATION SYSTEM SECTION
sampleData = pd.read_csv('spotifysampledata_0510.csv')
dataTypes = sampleData.dtypes
print(dataTypes)

# Feature engineering: convert genre strings into list
sampleData['genresList'] = sampleData['Genre'].apply(lambda x: [genre.strip() for genre in str(x).split(',') if isinstance(x, str)])
sampleData = sampleData.explode('genresList')
sampleData.pop('Genre')
# Sort by popularity in descending order and get the top 10 genres
sampleDataGenres = sampleData['genresList']
topGenres = sampleData.groupby('genresList')['Popularity'].mean().reset_index()
top10Genres = topGenres.sort_values(by = 'Popularity', ascending = False).head(10)
print(top10Genres)
# sampleData.to_csv('updatedSpotifySampleData15.csv')

# Recommendation system
# 1. Creating a playlist vector
def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
        
    Returns: 
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe): 
    """
    
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
    
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist

# 2. Generate playlist recommendations using cosine similarity
def generate_playlist_recos(df, features, nonplaylist_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """

    # Authentication key
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(), requests_timeout = 10, retries = 10)
    # Recommendation system
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    # non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(lambda x: spotify.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_40


