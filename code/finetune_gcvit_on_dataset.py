import os
import sys
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import time
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from EyeDataset import EyeDataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GCVit.models import gc_vit_large

# Import your helper modules
from helpers.visualize_dataset import visualize_samples, check_class_distribution
from helpers.graph_helpers import plot_loss, plot_confusion_matrix, plot_accuracy, plot_class_distribution, plot_ROC_AUC, calculate_metrics, plot_sensitivity_specificity
from helpers.extract_features import extract_features
from helpers.focal_loss import FocalLoss
from helpers.early_stopping import EarlyStopping

def reset_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # =======================
    # SETTINGS & SEED RESET
    # =======================
    reset_random_seeds(42)

    # =======================
    # DATASET & TRANSFORMS
    # =======================
    img_width, img_height, batch_size, epochs, mixup_alpha, cutmix_alpha, max_grad_norm = 224, 224, 16, 50, 1.0, 1.0, 1.0

    p_mix_start,  p_mix_end  = 0.05, 0.5
    p_cutmix_start, p_cutmix_end = 0.05, 0.3
    ramp_epochs = 2

    train_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.5048, 0.5281, 0.5611], std=[0.5048, 0.5281, 0.5611])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5048, 0.5281, 0.5611], std=[0.5048, 0.5281, 0.5611])
    ])

    data_path = './aptos-augmented-images'

    train_dataset = EyeDataset(root_dir=data_path, transform=train_transform)
    test_dataset = EyeDataset(root_dir=data_path, transform=test_transform)

    # Get image paths and labels from the train dataset for a stratified split
    all_image_paths = train_dataset.image_paths
    all_labels = train_dataset.labels

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    # Update the train and test dataset object with new splits
    train_dataset.image_paths = train_paths
    train_dataset.labels = train_labels

    test_dataset.image_paths = test_paths
    test_dataset.labels = test_labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # =======================
    # DERIVE CLASS NAMES FROM FOLDERS
    # =======================
    class_names = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
    train_dataset.classes = class_names
    test_dataset.classes = class_names

    # =======================
    # PLOT CLASS DISTRIBUTION
    # =======================
    train_counts = Counter(train_labels)
    # map integer label → folder name
    named_train_counts = { class_names[k]: v for k, v in train_counts.items() }
    plot_class_distribution(named_train_counts, title="Training Set Class Distribution")

    test_counts = Counter(test_labels)
    named_test_counts = { class_names[k]: v for k, v in test_counts.items() }
    plot_class_distribution(named_test_counts,  title="Validation Set Class Distribution")


    # Get number of classes from training dataset.
    num_classes = len(set(train_labels))
    print(f"Number of classes: {num_classes}")

    # =======================
    # MODEL: Fine-tuning ViT
    # =======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create the GCViT model without pretrained weights
    feature_extractor = gc_vit_large(pretrained=False, num_classes=num_classes, img_size=224)

    feature_extractor = feature_extractor.to(device)

    # Unfreeze all parameters for training
    for param in feature_extractor.parameters():
        param.requires_grad = True

    # Define optimizer with proper parameter grouping
    head_params = []
    backbone_params = []

    for name, param in feature_extractor.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # Define optimizer and loss
    optimizer = optim.Adam([
        {'params': [p for level in feature_extractor.levels for block in level.blocks for p in block.parameters()], 'lr': 1e-5},  # Parameters in all blocks
        {'params': feature_extractor.head.parameters(), 'lr': 1e-4}  # Head parameters
    ], weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # =======================
    # TRAINING LOOP
    # =======================
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_path = './models/best_gcvit_clean.pth'
    early_stopping = EarlyStopping(patience=5, verbose=True, mode='max')

    print("Starting fine-tuning...")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        feature_extractor.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Epoch {epoch+1}/{epochs}")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = feature_extractor(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_grad_norm)
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            train_loss += loss.item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        feature_extractor.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for val_imgs, val_labels in test_loader:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_outputs = feature_extractor(val_imgs)
                v_loss = criterion(val_outputs, val_labels)

                val_loss += v_loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                correct_val += (val_preds == val_labels).sum().item()
                total_val += val_labels.size(0)

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(feature_extractor.state_dict(), best_model_path)
            print(f"*** New best model saved (Val Acc={best_val_acc:.4f}) ***")

        # Check early stopping
        early_stopping(epoch_val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # --- Combined Plots for Loss & Accuracy ---
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Train vs Val Loss'); plt.legend(); plt.grid(); plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Acc')
    plt.plot(range(1, epochs+1), val_accuracies,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Train vs Val Accuracy'); plt.legend(); plt.grid(); plt.show()

    # Plot training and validation loss
    plot_loss(epochs, train_losses, title="Training Loss")
    plot_loss(epochs, val_losses, title="Validation Loss")

    # Plot training and validation accuracy
    plot_accuracy(epochs, train_accuracies, title="Training Accuracy")
    plot_accuracy(epochs, val_accuracies, title="Validation Accuracy")

    # =======================
    # CONFUSION MATRIX
    # =======================
    print("Generating confusion matrix on test set...")
    all_preds = []
    all_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = feature_extractor(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    else:
        class_names = sorted(set(all_targets))

    plot_confusion_matrix(all_targets, all_preds, class_names=class_names, title="Confusion Matrix")

    plot_ROC_AUC(feature_extractor, test_loader, num_classes)

    sensitivity, specificity = calculate_metrics(model=feature_extractor, dataloader=test_loader, device=device)
    plot_sensitivity_specificity(sensitivity, specificity, class_labels=class_names)

    # =======================
    # FEATURE EXTRACTION
    # =======================
    print("Extracting features using the fine-tuned ViT...")
    feature_extractor.load_state_dict(torch.load(best_model_path))
    feature_extractor.eval()
    with torch.no_grad():
        X_train, y_train = extract_features(train_loader, feature_extractor, device)
        X_test, y_test = extract_features(test_loader, feature_extractor, device)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =======================
    # SAVE FEATURES TO DISK
    # =======================
    feature_dir = './data/features/all_dataset_gcvit/'
    os.makedirs(feature_dir, exist_ok=True)
    train_feat_path = os.path.join(feature_dir, 'gcvit_X_train.pkl')
    test_feat_path = os.path.join(feature_dir, 'gcvit_X_test.pkl')

    joblib.dump((X_train, y_train), train_feat_path)
    joblib.dump((X_test, y_test), test_feat_path)
    print(f"Saved training features to: {train_feat_path}")
    print(f"Saved test features to: {test_feat_path}")
