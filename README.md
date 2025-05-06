# EC-523-Speech-to-Emotion-Recognition
# Introduction:

Our task is to use a deep learning architecture to identify the underlying emotion given some English speaking audio, formally known as Speech to Emotion Recognition (SER). We would like our model to differentiate between 6 emotions: anger, disgust, fear, happiness, sadness, and neutrality.

**Requirements**:
- Python 3.10.16
  
**Packages**:
- librosa 
- matplotlib
- mamba-ssm
- numpy
- pandas
- pytorch-lightning
- pytorch-metric-learning
- random
- scipy
- seaborn
- soundfile
- torch
- torch-audiomentations
- torch-pitch-shift
- torchaudio
- torchmetrics
- torchvision
- torchvision (duplicate)
- transformers

Installed in conda environment using: 

`conda create -n yourenv pip`

`pip install -r requirements.txt`

### How to run: 

Launch a Jupyter notebook with the following settings:
- List of modules to load: miniconda, python, torch, and tensorflow
- Pre-Launch Command: conda activate *your_project_location*/envs/mamba-env/
- 5 cores minimum, but you can lower the number of workers 


## Dataset
Our project uses four different datasets CREMA-D, RAVDESS, TESS, and SAVEE. all input files are in WAV. 

- **[CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)](https://www.kaggle.com/datasets/ejlok1/cremad)**:  
  This dataset includes 7,442 audio clips from 91 actors (48 male, 43 female). Each actor spoke 12 sentences representing six emotions (Anger, Disgust, Fear, Happy, Neutral, Sad), expressed at four levels (Low, Medium, High, Unspecified).

- **[RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)**
  RAVDESS consists of 7,356 recordings from 24 professional actors (12 male, 12 female) who spoke two statements each. The emotions expressed in this dataset include calm, happy, sad, angry, fearful, surprise, and disgust, both at two levels of emotional intensity, along with two levels of neutral expression.
  
- **[TESS (Toronto Emotional Speech Set)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)**
  This dataset features 1,400 recordings of two female actors, ages 24 and 64, articulating 200 target words. Each word was spoken to convey one of seven emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral. 

- **[SAVEE (Surrey Audio-Visual Expressed Emotion)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)**
  SAVEE consists of 480 English statements recorded from 4 male actors. Each actor expressed 7 emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. 

## Results

The table below shows the test accuracies for all our of our models to be above 70%. The Mamba-CNN model has the highest accuracy of 74.29%. Majority of the models began overfit around 10 epochs but still uphold decent accuracy. 

| Model  | Test Accuracy (%) | Training Accuracy (%) |
| ------------- | ------------- | ------------- |
| Base CNN  | 73.5  | 99.55  |
| CNN with Gru  | 71.11  | 99.98  |
| CNN-Transformer  | 70.8  | 99.11  |
| ResNet  | 70.89  | 95.53  |
| Mamba  | 72.79  | 98.62  |
| Mamba CNN | 74.29 | 81.95 | 

| Dataset  | Test Accuracy (%) | Training Accuracy (%) |
| ------------- | ------------- | ------------- |
| CREMA-D  | 65.82  | 99.24  |
| RAVDESS  | 79.25  | 99.76  |
| TESS  | 100  | 100  |
| SAVEE  | 72.62  | 90.18  |

## Citation
-  T. V. L. Trinh, D. T. L. T. Dao, L. X. T. Le, and E. Castelli, “Emotional speech recognition using deep neural networks,” Sensors (Basel), vol. 22, no. 4, p. 1414, Feb. 2022, doi: 10.3390/s22041414. 
- R. A. Khalil, E. Jones, M. I. Babar, T. Jan, M. H. Zafar, and T. Alhussain, "Speech emotion recognition using deep learning techniques: A review," IEEE Access, vol.7,pp.117327–117345,2019,doi:10.1109/ACCESS.2019.2936124.
- S. Han, F. Leng, and Z. Jin, “Speech emotion recognition with a ResNet-CNN-Transformer parallel neural network,” 2021 International Conference on Communications, Information System and Computer Engineering (CISCE), vol. 2021, pp. 803–807, Beijing, China, 2021, doi: 10.1109/CISCE52179.2021.9445906.
- R. R. Subramanian, Y. Sireesha, Y. S. P. K. Reddy, T. Bindamrutha, M. Harika, and R. R. Sudharsan, "Audio emotion recognition by deep neural networks and machine learning algorithms," 2021 International Conference on Advancements in Electrical, Electronics, Communication, Computing and Automation (ICAECA), Coimbatore, India, 2021, pp. 1–6, doi: 10.1109/ICAECA52838.2021.9675492.
- NeuroByte, “Speech Emotion Recognition with TensorFlow: A CNN & CRNN Guide,” NeuroByte, Jan. 19, 2025. Available: https://neurobyte.org/guides/speech-emotion-recognition-cnns-crnns-tensorflow/. 
- Kempner Institute. (n.d.). Repeat after me: Transformers are better than state space models at copying. Harvard University. Retrieved March 4, 2025, from https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/
- Albert Gu and Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint arXiv:2312.00752 (2023).
- S. Sharanyaa, T. J. Mercy and S. V.G, "Emotion Recognition Using Speech Processing," 2023 3rd International Conference on Intelligent Technologies (CONIT), Hubli, India, 2023, pp. 1-5, doi: 10.1109/CONIT59222.2023.10205935. 


# 

This project was done Spring 2025 for EC 523 Deep Learning at Boston University. 

Anish Sinha, James Knee, Nathan Strahs, Tyler Nguyen, Varsha Singh
