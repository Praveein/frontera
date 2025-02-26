import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import tensorflow as tf
import soundfile as sf
import resampy  # Import resampy

import params as yamnet_params
import yamnet as yamnet_model
import features  # Import features.py

# Initialize the Params class to access model parameters
params = yamnet_params.Params()

# Load YAMNet model and processor
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

# Load Wav2Vec2 model and processor
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base-960h', num_labels=521)
processor_wav2vec2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec2_model.to(device)

# Ensemble method: Averaging Probabilities from both models
def ensemble_average(yamnet_scores, wav2vec2_scores):
    """Combine predictions from YAMNet and Wav2Vec2 by averaging probabilities."""
    avg_scores = (yamnet_scores + wav2vec2_scores) / 2
    return np.argmax(avg_scores, axis=1)

def process_audio(file_name, yamnet, wav2vec2_model, processor_wav2vec2, params):
    """Load, preprocess the audio and make ensemble predictions."""
    # Load and preprocess the audio for YAMNet
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    waveform = wav_data / 32768.0  # Normalize to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and resample to the required sample rate
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)  # Resample to 16kHz

    # Ensure waveform is of size 160000 (pad or trim)
    max_length = 160000
    if waveform.shape[0] < max_length:
        # Pad the waveform if it's shorter than the required size
        waveform = np.pad(waveform, (0, max_length - waveform.shape[0]))
    else:
        # Trim the waveform if it's longer than the required size
        waveform = waveform[:max_length]

    # Preprocess audio for Wav2Vec2 (use Wav2Vec2Processor)
    inputs_wav2vec2 = processor_wav2vec2(waveform, return_tensors="pt", sampling_rate=params.sample_rate)

    # Ensure the input tensor has a batch dimension of 1 (for a batch of size 1)
    inputs_wav2vec2 = {key: val.squeeze(0).unsqueeze(0).to(device) for key, val in inputs_wav2vec2.items()}  # Add batch dimension correctly

    # Get YAMNet predictions (scores)
    yamnet_scores, _, _ = yamnet(waveform)
    yamnet_scores = np.mean(yamnet_scores, axis=0)  # Average the scores over time

    # Get Wav2Vec2 predictions (logits)
    wav2vec2_outputs = wav2vec2_model(**inputs_wav2vec2)
    wav2vec2_scores = torch.nn.functional.softmax(wav2vec2_outputs.logits, dim=-1).cpu().detach().numpy()

    # Check the shape of the wav2vec2_scores
    #print(f"Shape of Wav2Vec2 scores: {wav2vec2_scores.shape}")  # Should be [1, 521] for batch size 1

    # Combine the predictions using ensemble averaging
    final_predictions = ensemble_average(yamnet_scores, wav2vec2_scores)

    # Get the top 5 predictions
    top5_i = np.argsort(final_predictions)[::-1][:5]
    print(f"{file_name} predictions:\n" + 
          '\n'.join(f'  {yamnet_classes[i]:12s}: {final_predictions[i]:.3f}' for i in top5_i))

# Example usage
file_name = 'test.wav'  # Replace with your audio file
process_audio(file_name, yamnet, wav2vec2_model, processor_wav2vec2, params)
