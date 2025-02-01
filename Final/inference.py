import os
import cv2
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch import nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def load_test_data(test_dir, transform):
    filenames, images = [], []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        if transform:
            img = transform(img)
        images.append(img)
        filenames.append(os.path.splitext(img_name)[0])
    return filenames, torch.stack(images)


def bagging_inference(weight_files, test_dir='./data/Images/test'):
    print(f"[INFO] Loading model weights from {len(weight_files)} models...")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for weight_file in weight_files:
        print(f"  Loading weights from '{weight_file}'...")
        model = ResNet18(num_classes=7)
        model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
    print("[INFO] All models loaded successfully.")

    print(f"[INFO] Loading test data from '{test_dir}'...")
    filenames, X_test = load_test_data(test_dir, transform)
    X_test = X_test.to(device)
    print(f"[INFO] Loaded {len(filenames)} test images.")

    print("[INFO] Starting Bagging inference...")
    predictions = []
    with torch.no_grad():
        for i, img in enumerate(X_test):
            img = img.unsqueeze(0)
            ensemble_outputs = []
            for model in models:
                output = model(img)
                ensemble_outputs.append(output.softmax(dim=1))

            avg_output = torch.mean(torch.stack(ensemble_outputs), dim=0)
            pred = avg_output.argmax(dim=1).item()
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(X_test)} images.")

    submission = pd.DataFrame({
        'filename': filenames,
        'label': predictions
    })
    submission_file = 'submission_bagging.csv'
    submission.to_csv(submission_file, index=False)
    print(f"[INFO] Submission file '{submission_file}' generated successfully.")

if __name__ == "__main__":
    weight_files = [
        './resnet18_bagging_1.pth',
        './resnet18_bagging_2.pth',
        './resnet18_bagging_3.pth',
        './resnet18_bagging_4.pth',
        './resnet18_bagging_5.pth',
        './resnet18_bagging_6.pth',
        './resnet18_bagging_6.pth',
        './resnet18_bagging_6.pth'
    ]

    weight_files = [wf for wf in weight_files if os.path.exists(wf)]
    if not weight_files:
        raise FileNotFoundError("No weight files found for Bagging models.")

    bagging_inference(weight_files)
