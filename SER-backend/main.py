from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
import tempfile
from utils import extract_mfcc, label_map
from model import Base_CNN
import os
import tempfile

app = FastAPI()

# CORS (for dev allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Base_CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await file.read())

        input_tensor = extract_mfcc(path).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()

        return {"emotion": label_map[predicted_idx]}
    finally:
        os.remove(path)