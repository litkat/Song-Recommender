import pandas as pd
import numpy as np
import random
import pickle

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import getpass

import warnings
warnings.filterwarnings('ignore')

# Get API keys from user
cl_id = getpass.getpass('Enter client_id: ')
cl_secret = getpass.getpass('Enter client_secret: ')
# Get credentials
sp = spotipy.Spotify(auth_manager = SpotifyClientCredentials(client_id = cl_id, client_secret = cl_secret))

# Load dataframes
df_top100 = pd.read_csv('data_hot_100.csv')
df_clusters = pd.read_csv('data_clusters.csv')

# Defining main functions:
def get_features(song_title, artist_name):
    # Connect to spotify
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cl_id, client_secret=cl_secret))
    # Search for the song and limit to 1 best match
    search_result = sp.search(q = song_title + ' artist:' + artist_name, type='track', limit=1)
    # Get URI
    uri = search_result["tracks"]["items"][0]['id']
    # Get song featuresd from Spotify
    features = sp.audio_features(uri)[0]
    return features

    
def get_from_top100(song_title, artist_name, data):
    # This function needs title, artist, and top100 dataframe
    # Set initial variables
    rec_title, rec_artist = song_title, artist_name
    # Ganerate random index
    random_index = random.choice(range(data.shape[0]))
    # Repeat search until song is not the same
    while ( rec_title.lower() == song_title ) & ( rec_artist.lower() == artist_name ):
        rec_title = data.loc[random_index]['song']
        rec_artist = data.loc[random_index]['artist']
    print('Recommended song is:\nArtist: {}\nTitle: {}'.format(rec_artist, rec_title))

def get_from_clusters(song_title, artist_name, data):
    # Get user song features from Spotify
    song_features = get_features(song_title, artist_name)
    # Create dataframe
    df_features = pd.DataFrame.from_dict(song_features, orient='index').transpose()
    # Select only useful features
    song_features = df_features[['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 
                                 'instrumentalness', 'liveness', 'valence', 'tempo']]
    # Convert all numbers to float
    for i in song_features.columns:
        song_features[i] = song_features[i].astype(float)
    
    # Load models
    scaler = pickle.load(open('StandardScaler.pkl','rb'))
    kmeans = pickle.load(open('kmeans.pkl','rb'))
    # Find matching cluster
    user_song_scaled = scaler.transform(song_features) 
    user_song_cluster = kmeans.predict(user_song_scaled)
    # Select songs from the cluster
    clust_ = data[data['cluster'] == list(user_song_cluster)[0]]
    # Get random song from the cluster
    random_song = random.choice(range(clust_.shape[0]))
    title = clust_.iloc[random_song]['song']
    artist = clust_.iloc[random_song]['artist']
    print('Recommended song from Spotify tracklist:\nArtist: {}\nTitle: {}'.format(artist, title))


# Ask user for song and artist
user_song = str(input('Enter song: ')).lower()
user_artist = str(input('Enter artist: ')).lower()

# If song is in "top100" dataframe, recommend something from "top100", else recommend from the matching cluster

if len(df_top100[(df_top100['song'].str.lower() == user_song) & (df_top100['artist'].str.lower() == user_artist)]) != 0:
    
    get_from_top100(user_song, user_artist, df_top100)
    
else:
    get_from_clusters(user_song, user_artist, df_clusters)