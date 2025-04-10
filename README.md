# EC-523-Speech-to-Emotion-Recognition
# Introduction:

Our task is to use a deep learning architecture to identify the underlying emotion given some English speaking audio, formally known as Speech to Emotion Recognition (SER). We would like our model to differentiate between anger, disgust, fear, happiness, sadness, and neutrality.

**Requirements**:
- Python 3.10.16
  
**Packages**:
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
- List of modules to load: miniconda
- Pre-Launch Command: conda activate *your_project_location*/envs/mamba-env/

  

## Dataset
Our project uses the [Crowd-Sourced Emotional Multimodal Actors Dataset ]([url](https://www.kaggle.com/datasets/ejlok1/cremad))(CREMA-D) to train and test models. In this dataset, there are 7442 audio clips from 91 actors, with 48 of them being male and 43 being female. Each actor spoke 12 sentences from six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad), with four different emotion levels (Low, Medium, High, unspecified). 

## Results
| Model  | Test Accuracy (%) | Training Accuracy (%) |
| ------------- | ------------- | ------------- |
| Base CNN  | 60.71  | 90.22  |
| CNN with Gru  | 59.57  | 99.93  |
| CNN-Transformer  | 63.13  | 90.86  |
| ResNet  | 59.57  | 91.45  |
| Mamba  | 53.59  | 50.81  |
| Mamba CNN |  |  | 
## Citation
- T. V. L. Trinh, D. T. L. T. Dao, L. X. T. Le, and E. Castelli, “Emotional speech recognition using deep neural networks,” Sensors (Basel), vol. 22, no. 4, p. 1414, Feb. 2022, doi: 10.3390/s22041414.
- R. A. Khalil, E. Jones, M. I. Babar, T. Jan, M. H. Zafar, and T. Alhussain, "Speech emotion recognition using deep learning techniques: A review," IEEE Access, vol.7,pp.117327–117345,2019,doi:10.1109/ACCESS.2019.2936124.
- S. Han, F. Leng, and Z. Jin, “Speech emotion recognition with a ResNet-CNN-Transformer parallel neural network,” 2021 International Conference on Communications, Information System and Computer Engineering (CISCE), vol. 2021, pp. 803–807, Beijing, China, 2021, doi: 10.1109/CISCE52179.2021.9445906.
- R. R. Subramanian, Y. Sireesha, Y. S. P. K. Reddy, T. Bindamrutha, M. Harika, and R. R. Sudharsan, "Audio emotion recognition by deep neural networks and machine learning algorithms," 2021 International Conference on Advancements in Electrical, Electronics, Communication, Computing and Automation (ICAECA), Coimbatore, India, 2021, pp. 1–6, doi: 10.1109/ICAECA52838.2021.9675492.
- Kempner Institute. (n.d.). Repeat after me: Transformers are better than state space models at copying. Harvard University. Retrieved March 4, 2025, from https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/
- Albert Gu and Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint arXiv:2312.00752 (2023).

# 

This project was done Spring 2025 for EC 523 Deep Learning at Boston University. 

Anish Sinha, James Knee, Nathan Strahs, Tyler Nguyen, Varsha Singh
