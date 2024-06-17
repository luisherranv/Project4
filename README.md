# Project 4 - Neural Network Model for Song Popularity Prediction using Spotify Data 
**Author**: Luis Herran

  
## Introduction

**As an artist, who is about to release an entirely new album, how can you predict which song might be the most successful one to release as a single?** 


Every song that is in the Spotify database has a series of metadata or audio features that are unique to each track. These features are usually specific to the song, such as the key or song duration, as well as ratios such as danceability and energy of the song. Spotify also provides a song's popularity, currently, from 0 to 100, with 0 being a random song by a very small artist with 3 followers recorded in an iPhone, to 100 being Flowers by Miley Cyrus. The goal of this model is to use the audio-specific features, as well as the artist's current popularity to predict how succesful the song will be in the charts, as determined by the song popularity. Neural networks will be used as the machine learning platform, and an interactive app using Python and Flask will be deployed for stakeholder uses. The [original dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks#) was provided by Amitansh Joshi in Kaggle, and it includes data for over 1 million tracks, however furhter data was collected using the Spotipy API in Python. 

  
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
### Initial Model:

Python v3.1 in GoogleColab was used throughout this project to collect data and train the models. The dataset was directly pulled from Kaggle using the **kaggle API**. All of the training and data prep for the neural network model was done using **tensorflow** and **sklearn**. Data handling was done using **pandas** and **numpy**.

The following steps were taken to create an initial model and determine the areas where optimizations might be necessary:
1. Unecessary columns such as `year` in which the track was release, `duration_ms` and `track_id` were dropped, as well as `artist_name` and `track_name`. The latter, although could have a potential impact on the training, were categorical data and with such a large data set would create a very complex model. The remaining columns are shown in the image below. <img width="1354" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/245971d5-0379-4bc3-9dc4-de62de00f544">
2. The `popularity` columns which will become the target array was then binned in ranges of 10. Meaning, the final output should be any of the 11 targets (e.g. 0-10, 10-20 ... 90-100). The ranges were identified with a value from 0 - 11. <img width="1611" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/933f86e7-3094-4653-8731-9f63e83a846b">
3. The `genre` column was the "dummied" since it is the only categorical column in the data set. <img width="1501" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/41440ad0-4f09-491f-882a-76d958b45adf">
4. The data was the split into features and target `popularity` and further into training and test data. The overall dataset was scaled in order to train the model. <img width="853" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/b43e993f-4ad7-4376-8ae4-9729d5451e53">
5. The model was created, including 4 layers, compiled and then trained (epoch=10) as show below. <img width="1025" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/d1e92cfd-09f1-4ee0-9e99-868f89842548">
6. The accuracy of the model was just 0.48. <img width="760" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/5b11aebd-a735-428c-a6f6-d3582db48089">

As a result of such low accuracy (just below the flip of a coin), further optimization was necessary. 

### Model Optimization:

After a series of iterations, the following steps were taken to optimize the model and increase accuracy. Each step was successful in increasing the accuracy respecitvely. The number of layers and neurons were also tested, and the optimal value was shown to be that of the initial model, those steps are not described below.
1. Collect Artist Popularity Data - The Spotipy API was used to collect suplemental data. The popularity of each of the artists in the dataset was collected separatelly, and using pandas, it was merged into the original dataframe. As expected, a song's popularity is highly influenced by the artist that sings it, and their current overall popularity. Including this additional data column (`artist_popularity`) was successful in increasing the accuracy to just above 0.50. <img width="988" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/f031806b-22ce-451a-b60d-fe1b67d8f69a">
2. Further Binning of Target - The intial model had 11 outputs (bins of 10) for the target (`popularity`). Further binning, to bins of 25, with a total of 4 outputs (popularity = 0-25, 25-50, 50-75, 75-100) was the next step. Although the functionality of the model decreased, since now it can only predict wether the song is expected to fall in 1 of the 4 quartiles of popularity, the accuracy increased significantly. <img width="927" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/587356cc-65cf-4256-a9fe-ad1a2cf6460b">
3. Re-introducing the Track Duration Column - By re-introducing the `duration_ms` column, originally dropped, it helped increase accuracy slightly.
4. Incrasing Epcoh to 50 <img width="1001" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/97849f22-6bcb-4b5b-8263-e759117d978a">

**The final accuracy of the [optimized model](https://github.com/luisherranv/Project4/blob/main/SpotifyModelOptimized_5.h5) was 0.7842!**

## App Development

An application using **Flask** was created to load the model, run it using input data, and predict the song popularity. The entire app [`app.py`](https://github.com/luisherranv/Project4/blob/main/app.py) is run through Python, using flask to update the html file [`index.html`](https://github.com/luisherranv/Project4/blob/main/templates/index.html), from which the data inputs are collected. 


The intial interface has text boxes to input the song's audio features as well as the popularity of the artist. <img width="1223" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/aa4cd6cb-5048-4e14-a1c8-2988b6b1fa18">

Upon imputing all the values and clicking predict, the model is run and the output is displayed, along with the probabilities of the song falling within any of the 4 popularity quartiles and a bar plot representation. <img width="1224" alt="image" src="https://github.com/luisherranv/Project4/assets/150373234/a7f7b330-8cef-4742-a04b-344c9a44a66b">

## Conclusion
As is, the model is able to predict with a 78.4% accuracy the pobabilities of a single track of being currently popular on Spotify. This can be used by artist to make decisions on which single to release to ensure that the overall popularity of the album once released. It can also serve for artist to see how changes to a song, for instance more instrumental or less acoustic, can impact the overall popularity.

  
Further data however can be collected to increase the accuracy. For instance, another important meta-data that impacts a song's popularity is the lyrics. Potentially, by understanding the messaging (love, breakup, friendship, political, etc) or even most used words in the lyrics, could potentially help overcome this gap. Other areas that can increase accuracy is the artists' efforts in marketing said song, for instance, how much is it played on the radio, is the artist on tour, is this a TikToc song?.

## Repository Files Index
[`SpotifyModelOptimized_5.h5`](https://github.com/luisherranv/Project4/blob/main/SpotifyModelOptimized_5.h5): Tensorflow Model

  
[`SpotifyModelOptimized_5.ipynb`](https://github.com/luisherranv/Project4/blob/main/SpotifyModelOptimized_5.ipynb): Optimized Model Training Notebook

  
[`Spotipy.ipynb`](https://github.com/luisherranv/Project4/blob/main/Spotipy.ipynb): Artist Popularity Data Pull Notebook

  
[`app.py`](https://github.com/luisherranv/Project4/blob/main/app.py): Python File for App

  
templates/[`index.html`](https://github.com/luisherranv/Project4/blob/main/templates/index.html): HTML File for the App


[`scaler_5.pkl`](https://github.com/luisherranv/Project4/blob/main/scaler_5.pkl): Model's Scaler File to Scale Input Data


[`training_columns 2.txt`](https://github.com/luisherranv/Project4/blob/main/training_columns%202.txt): Columns in the Training Data (including dummies)

## References
[Kaggle API](https://drive.google.com/file/d/1Q1UZbj4qXImimLv91y4wkLKFnJ5YHnuR/view?usp=share_link)  

[Spotipy API](https://spotipy.readthedocs.io/en/2.24.0/)  

[Kaggle Dataset "Spotify_1Million_Tracks" by Amitansh Joshi](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks#)  

Further help to develop this model and app was obtained from classroom material, and troubleshooting with AI platforms (ChatGPT, Claude)







