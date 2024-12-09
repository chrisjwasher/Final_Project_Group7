import torch
import torch.nn as nn
import torchaudio
from transformers import (Wav2Vec2Model, AutoTokenizer, AutoModel,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, HubertModel)
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

class AudioTextEmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, transcripts, bert_tokenizer_path, label_mapping_path, max_text_length=128):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transcripts = transcripts
        self.max_text_length = max_text_length


        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)

        # Initialize tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])

        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process text
        text_encoding = self.text_tokenizer(
            self.transcripts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        # Squeeze the batch dimension to get a 1D tensor: [max_length]
        text_ids = text_encoding["input_ids"].squeeze(0)
        text_mask = text_encoding["attention_mask"].squeeze(0)


        return {
            "asr_features": waveform,
            "text_ids": text_ids,  # [max_length]
            "text_mask": text_mask,  # [max_length]
            "label": torch.tensor(self.labels[idx]),
        }


class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, num_emotions, bert_path):
        super().__init__()

        # Load models
        # Speech-to-text
        self.asr_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr_encoder.config.apply_spec_augment = False

        # SER
        self.ser_encoder = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.ser_encoder.config.apply_spec_augment = False

        # Text classification
        self.text_encoder = AutoModel.from_pretrained(bert_path)

        # Get dimensions
        self.ser_dim = self.ser_encoder.config.hidden_size
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
        for model in [self.asr_encoder, self.ser_encoder, self.text_encoder]:
            for param in model.parameters():
                param.requires_grad = False


    def forward(self, asr_features, text_ids, text_mask):
        # Process ASR features
        asr_output = self.asr_encoder(asr_features).last_hidden_state
        asr_pooled = asr_output.mean(dim=1)
        asr_projected = self.asr_projection(asr_pooled)

        # Process SER features
        ser_output = self.ser_encoder(asr_features).last_hidden_state
        ser_pooled = ser_output.mean(dim=1)
        ser_projected = self.ser_projection(ser_pooled)

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
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()

            # Move all batch elements to device
            asr_features = batch['asr_features'].to(device)
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(asr_features, text_ids, text_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                asr_features = batch['asr_features'].to(device)
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(asr_features, text_ids, text_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

            val_accuracy = 100.0 * correct_val / total_val
            val_accuracies.append(val_accuracy)


        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {train_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multimodal_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('multimodal_plot.png')

def load_data(audio_df, label_mapping_path, bert_token_path):

    emotions = sorted(audio_df['Emotion'].unique())
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

    with open(label_mapping_path, "w") as f:
        json.dump({"classes": emotions}, f)

    audio_df['Label'] = audio_df['Emotion'].map(emotion_to_idx)

    train_df, val_df = train_test_split(audio_df, test_size=0.2, random_state=42)

    train_dataset = AudioTextEmotionDataset(
        audio_paths=train_df["Path"].tolist(),
        labels=train_df["Label"].tolist(),
        transcripts=train_df["Transcription"].tolist(),
        bert_tokenizer_path=bert_token_path,
        label_mapping_path=label_mapping_path,
        max_text_length=128
    )

    val_dataset = AudioTextEmotionDataset(
        audio_paths=val_df["Path"].tolist(),
        labels=val_df["Label"].tolist(),
        transcripts=val_df["Transcription"].tolist(),
        bert_tokenizer_path=bert_token_path,
        label_mapping_path=label_mapping_path,
        max_text_length=128
    )

    return train_dataset, val_dataset


def collate_fn(batch):
    asr_features = [item["asr_features"] for item in batch]  # Each is [1, length]
    text_ids = [item["text_ids"] for item in batch]  # Each is [max_length]
    text_mask = [item["text_mask"] for item in batch]      # Each is [128]
    labels = [item["label"] for item in batch]

    # Determine the maximum length of asr_features in this batch
    max_asr_len = max(feat.size(1) for feat in asr_features)

    # Create a padded tensor for ASR features of shape [batch, max_asr_len]
    padded_asr_features = torch.zeros(len(asr_features), max_asr_len)

    for i, feat in enumerate(asr_features):
        # feat is [1, length], remove the [1, ...] dimension
        padded_asr_features[i, :feat.size(1)] = feat.squeeze(0)

    # Stack text tensors
    text_ids = torch.stack(text_ids, dim=0)
    text_mask = torch.stack(text_mask, dim=0) # [batch, 128]
    labels = torch.tensor(labels)

    return {
        "asr_features": padded_asr_features,  # [batch, max_length]
        "text_ids": text_ids,
        "text_mask": text_mask,
        "label": labels,
    }

def create_dataloader(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def precompute_transcriptions(df, audio_path_col="Path", sampling_rate=16000, device="cuda"):
    # Add audio transcription to dataframe
    # Initialize ASR model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().to(device)

    transcriptions = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing Transcripts"):
        audio_path = row[audio_path_col]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if not desired sampling rate
        if sample_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, sampling_rate)
            waveform = resampler(waveform)

        # Convert waveform to numpy
        waveform_np = waveform.squeeze(0).numpy()

        # Prepare input for the model
        inputs = processor(waveform_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        transcriptions.append(transcription)

    # Add the transcriptions to your DataFrame
    df["Transcription"] = transcriptions
    return df

def main():
    # Paths to your fine-tuned models
    bert_path = "./emotion_bert_finetuned"
    label_mapping_path = "./emotion_bert_finetuned/label_mapping.json"
    data = pd.read_csv("./MELD_train_sent_emo_final.csv")

    # Load label mapping to get number of emotions
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    num_emotions = len(label_mapping['classes'])

    data = precompute_transcriptions(data)

    train_dataset, val_dataset = load_data(
        audio_df=data,
        label_mapping_path=label_mapping_path,
        bert_token_path=bert_path
    )

    batch_size = 2
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset, batch_size=batch_size)


    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionClassifier(
        num_emotions=num_emotions,
        bert_path=bert_path
    ).to(device)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        device=device
    )


if __name__ == "__main__":
    main()