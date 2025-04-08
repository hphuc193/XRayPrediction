import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Dataset tùy chỉnh cho ảnh X-quang
class XRayDataset(Dataset):
    def __init__(self, image_paths, labels, all_labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.all_labels = all_labels
        
        # Transform cơ bản
        basic_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]
        
        # Thêm data augmentation nếu được yêu cầu
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ] + basic_transforms)
        else:
            self.transform = transforms.Compose(basic_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("L")
            image = self.transform(image)
            label = torch.zeros(len(self.all_labels))
            for lbl in self.labels[idx]:
                if lbl in self.all_labels:
                    label[self.all_labels.index(lbl)] = 1
            return image, label
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {self.image_paths[idx]}: {e}")
            return None

# Thêm Dropout vào model
def build_model(num_labels):
    model = models.densenet121(pretrained=True)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1024, num_labels)
    )
    return model

# Hàm huấn luyện lại với dữ liệu mới
def retrain_model(model, model_path, all_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    df = pd.read_csv("C:/XRayProject/retrain_data/data.csv")
    new_images = df["image"].tolist()
    new_labels = [eval(lbl) if isinstance(lbl, str) and lbl != "None" else [] for lbl in df["user_label"].fillna("[]").tolist()]
    
    retrain_dataset = XRayDataset(new_images, new_labels, all_labels, augment=True)
    retrain_loader = DataLoader(retrain_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(retrain_loader):
            if data is None:
                continue
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 5 == 4:
                print(f"Retraining Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(retrain_loader)}, Loss: {running_loss/5:.4f}")
                running_loss = 0.0
        
        avg_loss = running_loss / len(retrain_loader) if running_loss > 0 else running_loss
        print(f"Retraining Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")
    
    new_model_path = model_path.replace(".pth", "_retrained.pth")
    torch.save(model.state_dict(), new_model_path)
    print(f"Retrained model saved at: {new_model_path}")
    return new_model_path

# Phần huấn luyện chính
def main():
    train_dir = "C:/XRayProject/dataset/images_train"
    test_dir = "C:/XRayProject/dataset/images_test"
    csv_path = "C:/XRayProject/dataset/sample_labels.csv"

    df = pd.read_csv(csv_path)
    
    # Sử dụng danh sách nhãn cố định để đồng bộ với app.py
    all_labels = sorted([
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
    ])

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for idx, row in df.iterrows():
        img_name = row["Image Index"]
        labels = row["Finding Labels"].split(",") if pd.notna(row["Finding Labels"]) else ["No Finding"]
        # Chỉ giữ lại các nhãn có trong all_labels
        filtered_labels = [lbl for lbl in labels if lbl in all_labels]
        if not filtered_labels:
            filtered_labels = ["No Finding"]
        
        train_img_path = os.path.join(train_dir, img_name)
        test_img_path = os.path.join(test_dir, img_name)
        
        if os.path.exists(train_img_path):
            train_images.append(train_img_path)
            train_labels.append(filtered_labels)
        elif os.path.exists(test_img_path):
            test_images.append(test_img_path)
            test_labels.append(filtered_labels)
        else:
            print(f"Không tìm thấy ảnh: {img_name} trong cả hai thư mục train và test")

    total_train_size = len(train_images)
    train_size = int(0.85 * total_train_size)
    val_size = total_train_size - train_size

    indices = np.random.permutation(total_train_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    final_train_images = [train_images[i] for i in train_indices]
    final_train_labels = [train_labels[i] for i in train_indices]
    val_images = [train_images[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]

    print(f"Tập train: {len(final_train_images)} ảnh")
    print(f"Tập validation: {len(val_images)} ảnh")
    print(f"Tập test: {len(test_images)} ảnh")
    print(f"Tất cả nhãn: {all_labels}")

    train_dataset = XRayDataset(final_train_images, final_train_labels, all_labels, augment=True)
    val_dataset = XRayDataset(val_images, val_labels, all_labels, augment=False)
    test_dataset = XRayDataset(test_images, test_labels, all_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(all_labels))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Giảm learning rate

    num_epochs = 3  # Tăng số epoch
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            if data is None:
                continue
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5  # Tăng ngưỡng lên 0.5
            train_correct += (preds == targets).sum().item()
            train_total += targets.numel()

            if i % 10 == 9:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}")
                running_loss = 0.0

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue
                images, targets = data
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                preds = torch.sigmoid(outputs) > 0.5  # Tăng ngưỡng lên 0.5
                val_correct += (preds == targets).sum().item()
                val_total += targets.numel()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Vẽ biểu đồ
    os.makedirs("C:/XRayProject/plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("C:/XRayProject/plots/loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("C:/XRayProject/plots/accuracy_plot.png")
    plt.close()

    # Đánh giá trên tập test với cơ chế chọn nhãn chính
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in test_loader:
            if data is None:
                continue
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            # Chọn nhãn có xác suất cao nhất làm nhãn chính
            preds = (probs == probs.max(dim=1, keepdim=True)[0]).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    accuracy = (all_preds == all_targets).mean()

    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    model_path = "C:/XRayProject/models/model_v2.pth"
    torch.save(model.state_dict(), model_path)
    with open(model_path.replace(".pth", "_acc.txt"), "w") as f:
        f.write(str(accuracy))

if __name__ == "__main__":
    main()