import gradio as gr
import torch
from model import Base_CNN
from utils import extract_mfcc, label_map

model = Base_CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

def predict_emotion(audio_path):
    if not audio_path:
        return "No audio received."

    input_tensor = extract_mfcc(audio_path).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    return label_map[predicted_idx]

# Gradio UI
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload or Record 4s Audio"),
    outputs=gr.Text(label="Predicted Emotion"),
    title="🎤 Speech Emotion Recognition",
    description="Upload or record a 4-second audio clip. The model will classify the emotion."
)

demo.launch()
