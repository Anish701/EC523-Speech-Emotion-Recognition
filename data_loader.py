import os
import pandas as pd

def load_cremad(cremad_path):
    # Put the cremad directory in a list
    cremad = os.listdir(cremad_path)
    # Make a list for emotion labels and a list for path to audio files
    emotions = []
    paths = []
    # Loop through all the files and extract the emotion label and path
    for file in cremad:
        # Extract the emotion label from the file name
        emotion = file.split('_')[2]
        if emotion == 'SAD':
            emotion = 'sadness'
        elif emotion == 'ANG':
            emotion = 'anger'
        elif emotion == 'DIS':
            emotion = 'disgust'
        elif emotion == 'FEA':
            emotion = 'fear'
        elif emotion == 'HAP':
            emotion = 'happiness'
        elif emotion == 'NEU':
            emotion = 'neutral'
        elif emotion == 'SUR':
            emotion = 'surprise'
        else:
            emotion = 'Unknown'
        # Extract the path
        path = cremad_path + file
        # Append the emotion and path to their lists
        emotions.append(emotion)
        paths.append(path)
    # Create a dataframe from the lists
    cremad_df = pd.DataFrame(emotions, columns=['Emotion'])
    cremad_df['Path'] = paths
    # Inspect the dataframe
    return cremad_df

def load_ravdess(ravdess_path):
    # Put the ravdess directory in a list
    ravdess = os.listdir(ravdess_path)
    # Make a list for emotion labels and a list for path to audio files
    emotions = []
    paths = []
    # Loop through all the actor directories in audio_speech_actors_01-24
    for dir in ravdess:
        # Loop through all the files in each directory
        for file in os.listdir(ravdess_path + dir):
            # Extract the emotion label from the file name
            emotion = file.split('-')[2]
            if emotion == '01':
                emotion = 'neutral'
            elif emotion == '02':
                emotion = 'calm'
            elif emotion == '03':
                emotion = 'happiness'
            elif emotion == '04':
                emotion = 'sadness'
            elif emotion == '05':
                emotion = 'anger'
            elif emotion == '06':
                emotion = 'fear'
            elif emotion == '07':
                emotion = 'disgust'
            elif emotion == '08':
                emotion = 'surprise'
            else:
                emotion = 'Unknown'
            # Extract the path
            path = ravdess_path + dir + '/' + file
            # Append the emotion and path to their lists
            emotions.append(emotion)
            paths.append(path)
    # Create a dataframe from the lists
    ravdess_df = pd.DataFrame(emotions, columns=['Emotion'])
    ravdess_df['Path'] = paths
    # Inspect the dataframe
    return ravdess_df

def load_tess(tess_path):
    # Put the tess directory in a list
    tess = os.listdir(tess_path)
    # Make a list for emotion labels and a list for path to audio files
    emotions = []
    paths = []
    # Loop through all the audio file directories
    for dir in tess:
        # Loop through all the files in each directory
        for file in os.listdir(tess_path + dir):
            # Extract the emotion label from the file name
            emotion = file.split('.')[0]
            emotion = emotion.split('_')[2]
            if emotion == 'ps':
                emotion = 'surprise'
            elif emotion == 'sad':
                emotion = 'sadness'
            elif emotion == 'disgust':
                emotion = 'disgust'
            elif emotion == 'angry':
                emotion = 'anger'
            elif emotion == 'happy':
                emotion = 'happiness'
            elif emotion == 'neutral':
                emotion = 'neutral'
            elif emotion == 'fear':
                emotion = 'fear'
            else:
                emotion = 'Unknown'
            # Extract the path
            path = tess_path + dir + '/' + file
            # Append the emotion and path to their lists
            emotions.append(emotion)
            paths.append(path)
    # Create a dataframe from the lists
    tess_df = pd.DataFrame(emotions, columns=['Emotion'])
    tess_df['Path'] = paths
    # Inspect the dataframe
    return tess_df

def load_savee(savee_path):
    # Put the savee directory in a list
    savee = os.listdir(savee_path)
    # Make a list for emotion labels and a list for path to audio files
    emotions = []
    paths = []
    # Loop through all the files in the ALL directory
    for file in savee:
        # Separate the wav file name from the emotion label
        emotion = file.split('.')[0]
        # Extract the emotion label from the file name
        emotion = emotion.split('_')[1]
        # Exclude the numbers from the emotion label
        emotion = emotion[:-2]
        if emotion == 'a':
            emotion = 'anger'
        elif emotion == 'd':
            emotion = 'disgust'
        elif emotion == 'f':
            emotion = 'fear'
        elif emotion == 'h':
            emotion = 'happiness'
        elif emotion == 'n':
            emotion = 'neutral'
        elif emotion == 'sa':
            emotion = 'sadness'
        elif emotion == 'su':
            emotion = 'surprise'
        else:
            emotion = 'Unknown'
        # Extract the path
        path = savee_path + file
        # Append the emotion and path to their lists
        emotions.append(emotion)
        paths.append(path)
    # Create a dataframe from the lists
    savee_df = pd.DataFrame(emotions, columns=['Emotion'])
    savee_df['Path'] = paths
    # Inspect the dataframe
    return savee_df