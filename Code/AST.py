import numpy as np
import pandas as pd
import torch
import evaluate
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
    # Encode emotion labels
    label_encoder = LabelEncoder()
    #audio_dataframe['Emotion_Encoded'] = label_encoder.fit_transform(audio_dataframe['Emotions'])

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



    audio_dataframe['label_id'] = audio_dataframe['Emotions'].apply(lambda x: emotions.index(x))
    audio_dataframe = audio_dataframe.rename(columns={'Path': 'audio', 'label_id': 'labels'})
    audio_dataframe = audio_dataframe[['audio', 'labels']]


    dataset = Dataset.from_pandas(audio_dataframe, features=features)


    return dataset

dataset = create_audio_dataset(audio_dataframe)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels")


pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'
SAMPLING_RATE = feature_extractor.sampling_rate

feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
mean = []
std = []


for i, (audio_input, labels) in enumerate(dataset["train"]):
    cur_mean = torch.mean(dataset["train"][i][audio_input])
    cur_std = torch.std(dataset["train"][i][audio_input])
    mean.append(cur_mean)
    std.append(cur_std)
feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)
feature_extractor.do_normalize = True


def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

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
    ], p=0.8, shuffle=True)

    wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

# Apply the transformation to the dataset
dataset = dataset.rename_column("audio", "input_values")

# w/o augmentations on the test set
dataset["test"].set_transform(preprocess_audio, output_all_columns=False)



# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)
# Update configuration with the number of labels in our dataset
config.num_labels = num_labels
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}
# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
model.init_weights()


training_args = TrainingArguments(
    output_dir="./ast_classifier",
    learning_rate=5e-5,  # Learning rate
    num_train_epochs=10,  # Number of epochs
    per_device_train_batch_size=8,  # Batch size per device
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20,
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
