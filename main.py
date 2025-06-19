import os
import sys
import cv2
import torch
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image

from Tex2loc import Tex2loc
from GeoGessr import GeoGuessCountryClassifier
from tekstoinator import extract_text_from_video

def load_classifier_model(model_path, device, num_classes):
    model = GeoGuessCountryClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return preprocess(frame)

def predict_country_on_video(video_path, model, device, label_encoder):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        img_tensor = preprocess_frame(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predictions.append(pred.item())

    cap.release()

    most_common_idx = Counter(predictions).most_common(1)[0][0]
    most_common_country = label_encoder.classes_[most_common_idx]
    return most_common_country

def main():
    video_path = "input/video.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' nie istnieje!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używany device: {device}")

    print("1. Wykonywanie OCR na video...")
    extracted_text_list = extract_text_from_video(video_path, frame_interval=25)
    extracted_text = " ".join(extracted_text_list)
    print(f"Znaleziony tekst (fragment): {extracted_text[:500]}...")

    print("2. Analiza lokalizacji z tekstu...")
    tex2loc = Tex2loc(device=device.type) 
    location_info = tex2loc.get_location_info(extracted_text)
    print(f"Lokalizacja z tekstu: {location_info}")

    print("3. Ładowanie klasyfikatora obrazów...")
   
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("dataset/country_dataset.csv", header=None, names=['country', 'lat', 'lon', 'local_path'])

    counts = df['country'].value_counts()
    valid_countries = counts[counts >= 2].index
    df = df[df['country'].isin(valid_countries)]

    countries = sorted(df['country'].unique())

    le = LabelEncoder()
    le.fit(countries)

    model_path = "best_model.pt"
    model = load_classifier_model(model_path, device, num_classes=158)
    print("Model załadowany.")

    print("4. Predykcje na klatkach video...")
    most_common_country = predict_country_on_video(video_path, model, device, le)
    print(f"Najczęściej przewidywany kraj na video: {most_common_country}")

    print("\n--- Podsumowanie ---")
    print(f"Tekst OCR: {extracted_text[:500]}...")
    print(f"Lokalizacja tekstowa: Miasto: {location_info['city']}, Kraj: {location_info['country']}, Kontynent: {location_info['continent']}")
    print(f"Predykcja modelu na klatkach: {most_common_country}")


if __name__ == "__main__":
    main()