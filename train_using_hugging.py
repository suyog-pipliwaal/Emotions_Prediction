import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from transformers import Trainer, TrainingArguments,  ViTForImageClassification, ViTFeatureExtractor, default_data_collator
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
os.environ['WANDB_MODE'] = 'disabled'
import joblib
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def compute_metrics(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
def load_images(example):
    feature_ext = feature_extractor
    img_path = f"./Dataset/Dataset/Images/{example['Image_name']}"
    image = Image.open(img_path).convert("RGB")
    example["pixel_values"] = feature_ext(image, return_tensors="pt")["pixel_values"].squeeze(0)
    return example
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train an emotion classification model with ViT.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    train_csv           = './Dataset/Dataset/train.csv'

    df = pd.read_csv(train_csv)  # Ensure correct delimiter
    print(df)
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Emotion"])
    class_names = label_encoder.classes_
    
    # Save label encoder for later inference
    # 
    joblib.dump(label_encoder, "./using_vit/label_encoder.pkl")


    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    dataset = dataset.map(load_images)


    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(class_names),
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)}
    )

    training_args = TrainingArguments(
        output_dir="./using_vit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir="./vit_logs",
        logging_steps=100,
        report_to="none",  # Disable Weights & Biases logging
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator

        )
    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model("using_vit")