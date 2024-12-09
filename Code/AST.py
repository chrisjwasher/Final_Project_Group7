import numpy as np
import pandas as pd
import torch
import evaluate
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
from audiomentations import (Compose, AddGaussianSNR, GainTransition, Gain,
                             ClippingDistortion, TimeStretch, PitchShift)
from transformers import AutoProcessor
import torchaudio

audio_dataframe = pd.read_csv('Crema_data.csv')

def create_audio_dataset(audio_dataframe):

    #label_mappings = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    emotions = audio_dataframe['Emotions'].unique().tolist()
    #print(class_labels)
    #print(label_mappings)

    # Define features with audio and label columns
    class_labels = ClassLabel(names=emotions)
    features = Features({
        "audio": Audio(),
        "labels": class_labels
    })

    # map emotion labels to indices
    audio_dataframe['label_id'] = audio_dataframe['Emotions'].apply(lambda x: emotions.index(x))
    audio_dataframe = audio_dataframe.rename(columns={'Path': 'audio', 'label_id': 'labels'})
    audio_dataframe = audio_dataframe[['audio', 'labels']]

    # Construct dataset
    dataset = Dataset.from_pandas(audio_dataframe, features=features)

    return dataset

# Create dataset and split into train/test
dataset = create_audio_dataset(audio_dataframe)
#dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0, stratify_by_column='labels')


# Load pretrained Audio Spectrogram Transformer & feature extractor
pretrained_model = 'MIT/ast-finetuned-audioset-10-10-0.4593'
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
print(feature_extractor)
# we set normalization to False in order to calculate the mean + std of the dataset
feature_extractor.do_normalize = False
SAMPLING_RATE = feature_extractor.sampling_rate
model_input_name = feature_extractor.model_input_names[0]
mean = []
std = []

for sample in dataset:
    audio_array = sample['audio']['array']
    mean.append(audio_array.mean())
    std.append(audio_array.std())

feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)
feature_extractor.do_normalize = True # Enable normalization


def preprocess_audio(batch):
    wavs = [input_values["array"] for input_values in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors='pt')
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch


def preprocess_audio_augment(batch):
    audio_augmentations = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=20),
        Gain(min_gain_db=-6, max_gain_db=6),
        GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2),
        PitchShift(min_semitones=-4, max_semitones=4),
    ],
        p=0.8,
        shuffle=True
    )

    wavs = [audio_augmentations(input_values["array"], sample_rate=SAMPLING_RATE)
            for input_values in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt')
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    return output_batch

# Apply the transformation to the dataset
dataset = dataset.rename_column("audio", "input_values")
dataset.set_transform(preprocess_audio, output_all_columns=False)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0, stratify_by_column='labels')
print(dataset["train"].column_names)
print(dataset["train"][0])

# with augmentations on the training set
dataset["train"].set_transform(preprocess_audio_augment, output_all_columns=False)
# w/o augmentations on the test set
dataset["test"].set_transform(preprocess_audio, output_all_columns=False)

# Model Configuration
num_labels = len(dataset["train"].features["labels"].names)
label2id = {name: i for i, name in enumerate(dataset["train"].features["labels"].names)}
id2label = {i: name for name, i in label2id.items()}

# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = num_labels
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}

# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
model.init_weights()


training_args = TrainingArguments(
    output_dir="./ast_classifier",
    learning_rate=5e-5,  # Learning rate
    num_train_epochs=25,  # Number of epochs
    per_device_train_batch_size=16,  # Batch size per device
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20,
    save_total_limit=1
)

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")
AVERAGE = "macro" if config.num_labels > 2 else "binary"

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,  # Use the metrics function from above
)



trainer.train()
# Save the final model and feature extractor
model.save_pretrained("./ast_classifier")
feature_extractor.save_pretrained("./ast_classifier")

accuracy_values = []
epoch_values = []
for log_entry in trainer.state.log_history:
    if 'eval_accuracy' in log_entry and 'epoch' in log_entry:
        accuracy_values.append(log_entry['eval_accuracy'])
        epoch_values.append(log_entry['epoch'])

plt.figure(figsize=(8, 6))
plt.plot(epoch_values, accuracy_values, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.grid(True)
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()
plt.savefig('ast_plot.png')