import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
OUTPUT_DIR = os.getcwd()



class AudioSpecDataset(Dataset):
    def __init__(self, audio_paths, labels, sample_rate=16000, time_steps=224, n_mels=224, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.time_steps = time_steps
        self.augment = augment

        # Mel spectrogram transformation
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,  # Match ViT input size
            n_fft=2048,
            win_length=1024,
            hop_length=512,
            f_min=20,
            f_max=8000
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

        # Image transforms for ViT
        self.image_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _resize_spectrogram(self, spec, target_time_steps=224):
        """Resize spectrogram to have consistent time steps"""
        # Get current dimensions
        n_mels, cur_time_steps = spec.shape

        if cur_time_steps == target_time_steps:
            return spec

        # Interpolate along time axis
        spec = spec.unsqueeze(0)  # Add channel dimension for interpolation
        spec = torch.nn.functional.interpolate(
            spec,
            size=(n_mels, target_time_steps),
            mode='bilinear',
            align_corners=False
        )
        spec = spec.squeeze(0)  # Remove channel dimension
        return spec

    def _prepare_spectrogram(self, waveform):
        # Generate mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to decibel scale
        mel_spec = self.amplitude_to_db(mel_spec)

        mel_spec = self._resize_spectrogram(mel_spec, self.time_steps)

        # Normalize
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

        # Apply augmentations if enabled
        if self.augment:
            if torch.rand(1) < 0.5:
                mel_spec = self.time_masking(mel_spec)
            if torch.rand(1) < 0.5:
                mel_spec = self.freq_masking(mel_spec)

        if mel_spec.size(1) > 224:
            mel_spec = mel_spec[:, :224]
        elif mel_spec.size(1) < 224:
            # Pad if too short
            pad_amount = 224 - mel_spec.size(1)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))

        # Ensure correct shape and repeat to make 3 channels
        mel_spec = mel_spec.repeat(3, 1, 1)  # [3, n_mels, time]
        mel_spec = self.image_transforms(mel_spec)

        return mel_spec

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.audio_paths[idx])

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Prepare spectrogram
        spectrogram = self._prepare_spectrogram(waveform)

        return {
            'pixel_values': spectrogram,
            'label': torch.tensor(self.labels[idx])
        }

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', learning_rate=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values).logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='[Validation]'):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(pixel_values).logits
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_model.pth')

    return model

def load_data(dataframe):

    emotion_map = {
        'angry': 0,  # Angry
        'disgust': 1,  # Disgust
        'fear': 2,  # Fear
        'happy': 3,  # Happy
        'neutral': 4,  # Neutral
        'sad': 5  # Sad
    }

    audio_paths = dataframe['Path'].tolist()
    labels = [emotion_map[emotion] for emotion in dataframe['Emotions']]

    return audio_paths, labels, emotion_map

'''
def prepare_model_and_data(audio_paths, labels, num_emotions):
    # Initialize feature extractor


    # Create datasets
    train_dataset = AudioSpecDataset(
        train_audio_paths,
        train_labels,
        feature_extractor,
        augment=True
    )
    val_dataset = AudioSpecDataset(
        val_audio_paths,
        val_labels,
        feature_extractor,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model
    model = EmotionRecognitionTransformer(num_emotions)

    return model, train_loader, val_loader, feature_extractor
'''
def evaluate_model(model, test_loader, emotion_map, device='cuda'):
    """
    Evaluate model performance and print detailed metrics
    """


    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            outputs = model(input_values, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Get emotion names in correct order
    emotion_names = [emotion for emotion, idx in sorted(emotion_map.items(), key=lambda x: x[1])]

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_names))

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    df = pd.read_csv('Crema_data.csv')
    audio_paths, labels, emotion_map = load_data(df)
    num_emotions = len(emotion_map)

    from sklearn.model_selection import train_test_split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        audio_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print("Initializing feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    print("Creating datasets...")
    train_dataset = AudioSpecDataset(train_paths, train_labels, feature_extractor, augment=True)
    val_dataset = AudioSpecDataset(val_paths, val_labels, feature_extractor, augment=False)
    test_dataset = AudioSpecDataset(test_paths, test_labels, feature_extractor, augment=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("Initializing ViT model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = model.to(device)

    # Train model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        device=device,
        learning_rate=LR
    )

    # Evaluate on test set
    print("Evaluating model...")
    evaluate_model(trained_model, test_loader, emotion_map, device)

    # Save final model and emotion mapping
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'feature_extractor': feature_extractor,
        'emotion_map': emotion_map
    }, final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == '__main__':
    main()
