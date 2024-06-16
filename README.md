# Project 4 - Neural Network Model for Song Popularity Prediction using Spotify Data 

  
## Introduction

**As an artist, who is about to release an entirely new album, how can you predict which song might be the most successful one to release as a single?**

Every song that is in the Spotify database has a series or metadata or audio features that are unique to each track. These features are usually specific to the song, such as the key or song duration, as well as ratios such as danceability and energy of the song. Spotify also provides a song's popularity, currently, from 0 to 100, with 0 being a random song by a very small artist with 3 followers recorded in an iPhone, to 100 being Flowers by Miley Cyrus. The goal of this model is to use the audio-specific features, as well as the artist's current popularity to predict how succesful the song will be in the charts, as determined by the song popularity. Neural networks will be used as the machine learning platform, and an interactive app using Python and Flask will be deployed for stakeholder uses. The [original dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks#) was provided by Amitansh Joshi in Kaggle, and it includes data for over 1 million tracks, however furhter data was collected using the Spotipy API in Python. 

  
Below are the list of features used to train the model to predict song `popularity` from 0 through 100:
1. `danceability`: track's suitability for dancing (0 - 1)
2. `energy`: perceptual measure of intensity and activity (0 - 1)
3. `key`: key in which the track is in (-1 - 11)
4. `loudness`: overall loudness of the track in decibels (-60 to 0 dB)
5. `mode`: modality of the track (major = 1, minor = 0)
6. `speechines`: precense of spoken words in the track (0 - 1)
7. `acousticness`: confidence measure of whether the track is acoustic (0 - 1)
8. `instrumentalness`: whether the track contains vocals (0 - 1)
9. `liveness`: precense of audience in the recording (0 - 1)
10. `valence`: musical positiveness (0 - 1)
11. `tempo`: tempo of the track in beats per min (BPM)
12. `time_signature`: estimated time signature (3 - 7)
13. `duration_ms`: duration of track in milliseconds
14. `artist_popularity`: overall current popularity of the artist (0 - 100)

## Model Development
**Initial Model:**

