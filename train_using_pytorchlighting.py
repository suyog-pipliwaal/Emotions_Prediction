import torch
import sklearn
import pandas
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.models as models
import os
import pandas as pd
import pytorch_lightning as pl
os.environ['WANDB_MODE'] = 'disabled'
import matplotlib.pyplot as plt
import argparse
import joblib
from torchmetrics.classification import Accuracy, ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# torch.set_float32_matmul_precision('medium')
class EmotionDetectionDataset(torch.utils.data.Dataset):
    def __init__(self,df, root_dir, transform=None):
        self.dataframe = df
        self.root_dir = root_dir
        self.transform = transform
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[index,0])
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        # print(f"index is {index} and data {data}")
        return image, torch.tensor(label, dtype=torch.long)
    def __len__(self):
        return len(self.dataframe)
    


class EmotionalModel(pl.LightningModule):
    def __init__(self, num_class):
        super(EmotionalModel,self).__init__()
        # self.model = models.resnet18(weights='DEFAULT')
        # in_features = self.model.fc.in_features
        # self.model.fc = torch.nn.Linear(in_features, num_class)

        # self.model = models.efficientnet_b3(weight='DEFAULT')  # Using EfficientNet-B3
        # in_features = self.model.classifier[1].in_features  # Get input features of classifier
        # self.model.classifier[1] = torch.nn.Linear(in_features, num_class)
        self.cnn_model = torch.nn.Sequential(
            torch.nn.Conv2d(3,32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(64 * 53 * 53, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_class)
        )
     
     
        self.criterion = torch.nn.CrossEntropyLoss()
     
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        self.predictions = []
        self.true_labels = []

    def training_step(self, batch, batch_idx):
        image, label = batch
        outputs = self.forward(image)
        loss = self.criterion(outputs, label)
        self.log('train_loss', loss)
        # self.train_losses.append(loss.item())
        return loss

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fully_connected(x)
        return x

    def validation_step(self, batch, batch_idx):
        image, label = batch
        outputs = self.forward(image)
        loss = self.criterion(outputs, label)
        acc = (outputs.argmax(dim=1) == label).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        self.true_labels.extend(label.cpu().numpy())
        # self.val_losses.append(loss.item())
        # self.val_accuracies.append(acc.item())
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = self.trainer.logged_metrics.get('train_loss', torch.tensor(0.0)).item()
        avg_val_loss = self.trainer.logged_metrics.get('val_loss', torch.tensor(0.0)).item()
        avg_val_acc = self.trainer.logged_metrics.get('val_acc', torch.tensor(0.0)).item()
        self.train_losses.append(avg_train_loss)
        self.val_losses.append(avg_val_loss)
        self.val_accuracies.append(avg_val_acc)
        print(f'End of Epoch - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}')
    def on_train_end(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig('loss_curve.png')
        
        plt.figure(figsize=(10,5))
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig('./custom_model/accuracy_curve.png')


        cm = confusion_matrix(self.true_labels, self.predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(8,8))
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("./custom_model/confusion_matrix.png")

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an emotion classification model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_csv           = './Dataset/Dataset/train.csv'
    df                  = pd.read_csv(train_csv)
    label_encoder       = LabelEncoder()
    df['label']         = label_encoder.fit_transform(df['Emotion'])
    class_names         = label_encoder.classes_
    joblib.dump(label_encoder, "./custom_model/label_encoder.pkl")

    train_df, val_df    = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_transforms    = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
    
    val_transforms      = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    train_dataset       = EmotionDetectionDataset(train_df, './Dataset/Dataset/Images', train_transforms)
    train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=11)

    
    val_dataset         = EmotionDetectionDataset(val_df, './Dataset/Dataset/Images',val_transforms)
    val_loader          = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=11)
    model               = EmotionalModel(len(class_names))
    print(class_names) 
     
    # print(train_dataset.__getitem__(0)[0].shape)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="cpu",  default_root_dir="./custom_model")
    trainer.fit(model, train_loader, val_loader)