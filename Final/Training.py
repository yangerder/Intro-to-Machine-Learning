import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, val_accuracies, model_idx):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model {model_idx} - Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {model_idx} - Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'learning_curve_model_{model_idx}.png')

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_and_clean_data(data_dir, threshold=100):
    print(f"[INFO] Loading and cleaning data from {data_dir}...")
    image_paths = []
    labels = []
    emotions = os.listdir(data_dir)
    for label, emotion in enumerate(emotions):
        folder_path = os.path.join(data_dir, emotion)
        for img_path in glob.glob(f"{folder_path}/*.jpg"):
            if not is_blurry(img_path, threshold):
                image_paths.append(img_path)
                labels.append(label)
    print(f"[INFO] Loaded {len(image_paths)} images after filtering.")
    return image_paths, labels

class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def bootstrap_sampling(data, labels, n_samples):
    indices = np.random.choice(len(data), n_samples, replace=True)
    return [data[i] for i in indices], [labels[i] for i in indices]

def train_single_model(train_data, train_labels, val_data, val_labels, model_idx, device, num_classes=7,num_epochs=30):
    train_dataset = EmotionDataset(train_data, train_labels, transform=train_transform)
    val_dataset = EmotionDataset(val_data, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = ResNet18(num_classes=num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"[Model {model_idx}] Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_accuracy)

        print(f"[Model {model_idx}] Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    plot_learning_curve(train_losses, val_accuracies, model_idx)

    torch.save(model.state_dict(), f'resnet18_bagging_{model_idx}.pth')
    print(f"[Model {model_idx}] Training complete. Model weights saved.")
    return model

def train_bagging_models(image_paths, labels, val_paths, val_labels, device):
    num_bagging_models = 5
    models = []
    for i in range(num_bagging_models):
        print(f"[INFO] Training Bagging Model {i+1}/{num_bagging_models}...")
        train_data, train_labels = bootstrap_sampling(image_paths, labels, len(image_paths))
        model = train_single_model(train_data, train_labels, val_paths, val_labels, model_idx=i+1, device=device,num_epochs=30)
        models.append(model)

    print(f"[INFO] Training Bagging Model 6 (without bootstrap)...")
    model = train_single_model(image_paths, labels, val_paths, val_labels, model_idx=num_bagging_models+1, device=device,num_epochs=20)
    models.append(model)

    return models


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = './data/Images/train'
    image_paths, labels = load_and_clean_data(train_dir, threshold=100)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    bagging_models = train_bagging_models(train_paths, train_labels, val_paths, val_labels, device=device)

