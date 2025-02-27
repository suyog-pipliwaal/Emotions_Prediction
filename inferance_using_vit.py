import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from transformers import Trainer, TrainingArguments,  ViTForImageClassification, ViTFeatureExtractor, default_data_collator
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import joblib
from tqdm import tqdm

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
def predict_emotion(image_path, model, label_encoder):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(image, return_tensors="pt")  # Convert image to tensor format
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    logits = outputs.logits
    predicted_class = np.argmax(logits.numpy())  # Convert tensor to NumPy and get the class index

    # Convert class index back to emotion label
    emotion_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return emotion_label
if __name__ == '__main__':
    model_path          = './using_vit/'
    label_encoder       = joblib.load("./using_vit/label_encoder.pkl")
    model               = ViTForImageClassification.from_pretrained(model_path)
    feature_extractor   = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    test_csv            = './Dataset/Dataset/test.csv'
    transform           = Compose([Resize((224, 224)),ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    df                  = pd.read_csv(test_csv)
    

    ans = []
    for index, row in tqdm(df.iterrows()):
        if index == -1:
            break
        image_path =  os.path.join('./Dataset/Dataset/Images', row['Image_name'])
        predicted_emotion = predict_emotion(image_path, model, label_encoder)
        # print(predicted_emotion)
        ans.append(predicted_emotion)
    submission = pd.DataFrame({'Emotion':ans})
    submission.to_csv('./using_vit/submission.csv', index=False)




