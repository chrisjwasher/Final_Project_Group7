import torch
import torch.nn as nn
import torchaudio
from transformers import (Wav2Vec2ForCTC, AutoTokenizer, AutoModel,
                          Wav2Vec2FeatureExtractor, AutoModelForAudioClassification)
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np


class AudioTextEmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, bert_tokenizer_path, label_mapping_path):
        self.audio_paths = audio_paths
        self.labels = labels

        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)

        # Initialize processors and tokenizers
        self.asr_processor = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
        self.ser_processor = Wav2Vec2FeatureExtractor.from_pretrained(ser_path) #NEED SER MODEL
        self.text_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process audio for ASR
        asr_features = self.asr_processor(waveform, sampling_rate=16000).input_values

        # Process audio for SER
        ser_features = self.ser_processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values

        # Get transcription
        with torch.no_grad():
            logits = self.asr_processor(asr_features).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.decode(predicted_ids[0])

        # Process text
        text_encoding = self.text_tokenizer(
            transcription,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'asr_features': torch.tensor(asr_features),
            'ser_features': ser_features.squeeze(),
            'ser_attention_mask': ser_features.attention_mask.squeeze(),
            'text_ids': text_encoding['input_ids'].squeeze(),
            'text_mask': text_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }


class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, num_emotions, bert_path, ser_path):
        super().__init__()

        # Load models
        # Speech-to-text
        self.asr_encoder = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

        # SER
        self.ser_encoder = AutoModelForAudioClassification.from_pretrained(ser_path)

        # Text classification
        self.text_encoder = AutoModel.from_pretrained(bert_path)

        # Get dimensions
        self.ser_dim = self.ser_encoder.classifier.in_features
        self.bert_dim = self.text_encoder.config.hidden_size

        # Fusion and classification layers
        self.asr_projection = nn.Linear(768, 256)
        self.ser_projection = nn.Linear(self.ser_dim, 256)
        self.text_projection = nn.Linear(self.bert_dim, 256)

        # Enhanced classifier with attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),  # 768 = 256 * 3 (concatenated features)
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_emotions)
        )

        # Freeze pretrained models
        for model in [self.asr_encoder, self.ser_encoder]:
            for param in model.parameters():
                param.requires_grad = False


    def forward(self, asr_features, ser_features, text_ids, text_mask):
        # Process ASR features
        asr_output = self.asr_encoder(asr_features).last_hidden_state
        asr_pooled = asr_output.mean(dim=1)
        asr_projected = self.asr_projection(asr_pooled)

        # Process SER features with attention mask
        ser_output = self.ser_encoder(
            input_values=ser_features,
            attention_mask=ser_attention_mask
        ).logits
        ser_projected = self.ser_projection(ser_output)

        # Process text with fine-tuned BERT
        text_output = self.text_encoder(text_ids, attention_mask=text_mask).last_hidden_state
        text_pooled = text_output.mean(dim=1)
        text_projected = self.text_projection(text_pooled)

        # Apply attention to fuse modalities
        features = torch.stack([asr_projected, ser_projected, text_projected], dim=0)
        attended_features, _ = self.attention(features, features, features)

        # Concatenate attended features
        combined_features = attended_features.transpose(0, 1).reshape(asr_projected.size(0), -1)

        # Final classification
        logits = self.classifier(combined_features)

        return logits


def train_model(model, train_loader, val_loader, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move all batch elements to device
            asr_features = batch['asr_features'].to(device)
            ser_features = batch['ser_features'].to(device)
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(asr_features, ser_features, text_ids, text_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                asr_features = batch['asr_features'].to(device)
                ser_features = batch['ser_features'].to(device)
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(asr_features, ser_features, text_ids, text_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Print metrics
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {100. * correct / total:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multimodal_model.pth')


def main():
    # Paths to your fine-tuned models
    bert_path = "./emotion_bert_finetuned"
    ser_path = "./ser_model_finetuned"
    label_mapping_path = "./emotion_bert_finetuned/label_mapping.json"

    # Load label mapping to get number of emotions
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    num_emotions = len(label_mapping['classes'])

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionClassifier(
        num_emotions=num_emotions,
        bert_path=bert_path,
        ser_path=ser_path
    ).to(device)

    # Data loading
    train_loader =
    val_loader =

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        device=device
    )


if __name__ == "__main__":
    main()