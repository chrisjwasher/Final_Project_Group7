# fine_tune_bert.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import os

class TextEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_length=128):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx])
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return total_loss / len(dataloader), accuracy


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv('MELD_train_sent_emo_final.csv')
    texts = df['Utterance'].tolist()
    labels = df['Emotion'].tolist()

    # Initialize label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Save label encoder classes for later use
    label_mapping = {
        'classes': label_encoder.classes_.tolist(),
        'num_labels': len(label_encoder.classes_)
    }



    os.makedirs("./emotion_bert_finetuned", exist_ok=True)
    with open("./emotion_bert_finetuned/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f)

    print(f"Emotion classes: {label_mapping['classes']}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=label_mapping['num_labels']
    )
    model.to(device)


    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # Create datasets
    train_dataset = TextEmotionDataset(train_texts, train_labels, tokenizer, label_encoder)
    val_dataset = TextEmotionDataset(val_texts, val_labels, tokenizer, label_encoder)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    # Training settings
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            model.save_pretrained("./emotion_bert_finetuned")
            tokenizer.save_pretrained("./emotion_bert_finetuned")


if __name__ == "__main__":
    main()