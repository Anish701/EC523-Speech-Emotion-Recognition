import os
import pandas as pd

def load_cremad(cremad_path):
    cremad = os.listdir(cremad_path)
    emotions = []
    paths = []
    for file in cremad:
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

        path = cremad_path + file
        emotions.append(emotion)
        paths.append(path)

    cremad_df = pd.DataFrame(emotions, columns=['Emotion'])
    cremad_df['Path'] = paths
    return cremad_df

def load_ravdess(ravdess_path):
    ravdess = os.listdir(ravdess_path)
    emotions = []
    paths = []
    for dir in ravdess:
        for file in os.listdir(ravdess_path + dir):
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
            
            path = ravdess_path + dir + '/' + file
            emotions.append(emotion)
            paths.append(path)
    
    ravdess_df = pd.DataFrame(emotions, columns=['Emotion'])
    ravdess_df['Path'] = paths
    return ravdess_df

def load_tess(tess_path):
    tess = os.listdir(tess_path)
    emotions = []
    paths = []
    for dir in tess:
        for file in os.listdir(tess_path + dir):
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
            
            path = tess_path + dir + '/' + file
            emotions.append(emotion)
            paths.append(path)
    
    tess_df = pd.DataFrame(emotions, columns=['Emotion'])
    tess_df['Path'] = paths
    return tess_df

def load_savee(savee_path):
    savee = os.listdir(savee_path)
    emotions = []
    paths = []
    for file in savee:
        emotion = file.split('.')[0]
        emotion = emotion.split('_')[1]
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
        
        path = savee_path + file
        emotions.append(emotion)
        paths.append(path)
    
    savee_df = pd.DataFrame(emotions, columns=['Emotion'])
    savee_df['Path'] = paths
    return savee_df