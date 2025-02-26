import glob
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa
from collections import namedtuple

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Set to block CUDA errors

# Load the class map CSV (mapping from mid to index)
class_map_df = pd.read_csv('yamnet_class_map.csv')
class_map = pd.read_csv('yamnet_class_map.csv').set_index('display_name').to_dict()['mid']

# Create a mapping from mid (string) to index (integer)
mid_to_index = {mid: idx for idx, mid in enumerate(set(class_map.values()))}

# Initialize the model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base-960h', num_labels=521)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Moved model to GPU if available")

# Define a namedtuple for dataset items
AudioSample = namedtuple("AudioSample", ["input_values", "labels"])

class AudioDataset(Dataset):
    def __init__(self, audio_directory, ontology_file, mid_to_index):
        with open(ontology_file, 'r') as f:
            self.ontology_data = json.load(f)

        self.mid_to_index = mid_to_index
        self.audio_directory = audio_directory
        self.audio_files = glob.glob(os.path.join(self.audio_directory, '**', '*.wav'), recursive=True)
        
        # Populate the dataset by calling prepare_data
        self.data = self.prepare_data()

    def prepare_data(self):
        data = []
        for category in self.ontology_data:
            if "positive_examples" in category:
                category_name = category["name"]
                mid = category["id"]  # Get the mid for the current category

                # Use the mid to get the index from the mid_to_index
                if mid in self.mid_to_index:
                    label = self.mid_to_index[mid]  # Get the integer index as the label
                else:
                    label = -1  # Default to -1 if not found

                for audio_file in self.audio_files:
                    if category_name.lower() in audio_file.lower():
                        audio_file = audio_file.replace("\\", "/")
                        data.append({"audio": audio_file, "label": label})
        return data
    
    def load_audio(self, file_path):
        """Load and preprocess audio using Wav2Vec2Processor."""
        try:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"WAV file not found: {file_path}")

            # Load audio using librosa and resample to 16kHz
            audio_data, sr = librosa.load(file_path, sr=16000)
            print(f"Successfully loaded audio: {file_path}, shape: {audio_data.shape}, dtype: {audio_data.dtype}")

            # Preprocess using Wav2Vec2Processor
            inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
            processed_audio = inputs.input_values.squeeze(0)  # Remove batch dimension

            print(f"Processed audio shape: {processed_audio.shape}")

            # Return processed audio
            return processed_audio

        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

    def __getitem__(self, idx):
        """Get one item (audio, label) for the dataset."""
        sample = self.data[idx]
        audio_data = self.load_audio(sample["audio"])
        label = sample["label"]

        # Ensure audio_data is valid
        if audio_data is None:
            print(f"Error loading audio at index {idx}, returning dummy data.")
            return {"input_values": torch.zeros(1), "labels": torch.tensor(label, dtype=torch.long)}

        # Trim or pad audio to max_length
        max_length = 160000  # Set a max_length for padding/truncating
        if audio_data.shape[0] < max_length:
            padding = torch.zeros(max_length - audio_data.shape[0])
            audio_data = torch.cat([audio_data, padding])
        else:
            audio_data = audio_data[:max_length]

        print(f"Successfully loaded audio at index {idx}, shape: {audio_data.shape}, label: {label}")
        
        return {"input_values": audio_data.clone().detach(), "labels": torch.tensor(label, dtype=torch.long)}

    def __len__(self):
        return len(self.data)

# Initialize the dataset and dataloaders
audio_directory = r"audiosets/ontology"
ontology_file = 'ontology.json'

# Initialize dataset and prepare data
dataset = AudioDataset(audio_directory, ontology_file, mid_to_index)

# Now split the dataset into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

# Initialize train and test datasets using the split data
train_dataset = AudioDataset(audio_directory, ontology_file, mid_to_index)
test_dataset = AudioDataset(audio_directory, ontology_file, mid_to_index)

# Assign the split data to the datasets
train_dataset.data = train_data
test_dataset.data = test_data

# Define the compute_metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# Training setup
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the model will be saved
    evaluation_strategy="steps",  # Save after each epoch
    save_strategy="steps",  # Save after each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=3,
    save_steps=500,
    disable_tqdm=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

# Initialize train and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Pass the datasets without specifying the dataloaders in Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# Training loop with handling for -1 labels
for epoch in range(int(training_args.num_train_epochs)):
    print(f"Training epoch {epoch + 1}")
    model.train()  # Set model to training mode
    for batch in train_dataloader:
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)

        # Skip batches where label is -1
        valid_indices = labels != -1
        if valid_indices.sum() == 0:
            continue  # Skip this batch if all labels are -1

        # Only select valid indices
        input_values = input_values[valid_indices]
        labels = labels[valid_indices]

        # Forward pass
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        loss.backward()  # Backward pass to compute gradients

        # Clear GPU cache to prevent memory overflow
        torch.cuda.empty_cache()
    # Manually save the model after each epoch
    model.save_pretrained(f"./results/checkpoint-{epoch+1}")
    print(f"Model saved at checkpoint-{epoch+1}")
    
    # Run evaluation after each epoch (optional)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in test_dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            # Skip batches where label is -1
            valid_indices = labels != -1
            if valid_indices.sum() == 0:
                continue  # Skip this batch if all labels are -1

            # Only select valid indices
            input_values = input_values[valid_indices]
            labels = labels[valid_indices]

            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            print(f"Evaluation loss: {loss.item()}")

    torch.cuda.empty_cache()  # Clear GPU cache after each epoch