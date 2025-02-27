import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import joblib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision.models as models
import pandas as pd
import os
class EmotionalModel(pl.LightningModule):
    def __init__(self, num_class):
        super(EmotionalModel,self).__init__()
     
     
        self.model = models.efficientnet_b3(weights='DEFAULT')
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_class)
     
     
        self.criterion = torch.nn.CrossEntropyLoss()
     
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def training_step(self, batch, batch_idx):
        image, label = batch
        outputs = self.model(image)
        loss = self.criterion(outputs, label)
        self.log('train_loss', loss)
        # self.train_losses.append(loss.item())
        return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        image, label = batch
        outputs = self.model(image)
        loss = self.criterion(outputs, label)
        acc = (outputs.argmax(dim=1) == label).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
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
        # avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        # avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        # avg_val_acc = sum(self.val_accuracies) / len(self.val_accuracies)
        # self.train_losses.append(avg_train_loss)
        # self.val_lossesa.apppend(avg_val_acc)
        # self.val_accuracies.append(avg_val_acc)
        # print(f'End of Epoch - Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_acc}')
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
        plt.savefig('accuracy_curve.png')

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_csv  = './Dataset/Dataset/test.csv'

    label_encoder = joblib.load("label_encoder.pkl")
    model = EmotionalModel.load_from_checkpoint("./using_efficientnet_b3/lightning_logs/version_3/checkpoints/epoch=34-step=6230.ckpt", num_class=len(label_encoder.classes_))
    model.eval()
    model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    prediction = list()
    df = pd.read_csv(train_csv)
    for index, row in df.iterrows():
        if index == -1:
            break
        image_path =  os.path.join('./Dataset/Dataset/Images', row['Image_name'])
        # true_label = row['Emotion']
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image.to(device))
            predicted_class = torch.argmax(output,dim=1).item()
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        prediction.append(predicted_emotion)
        print(f"index: {index} file name : {row['Image_name']} Predicted Emotion: {predicted_emotion}")
    submission = pd.DataFrame({'Emotion':prediction})
    submission.to_csv('./submission.csv', index=False)