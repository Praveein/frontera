import asyncio
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
import numpy as np
import torch
import librosa
import soundfile as sf
import resampy
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import yamnet as yamnet_model
import params as yamnet_params
import pandas as pd 

# Initialize the Params class to access model parameters
params = yamnet_params.Params()
YAMNET_PATH = "yamnet.h5"
class_map = pd.read_csv('yamnet_class_map.csv').set_index('display_name').to_dict()['index']

# Load YAMNet model and processor
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

# Load the Wav2Vec2 model and processor
model_path = r"results\checkpoint-1"
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor_wav2vec2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec2_model.to(device)

# Ensemble method: Averaging Probabilities from both models
def ensemble_average(yamnet_scores, wav2vec2_scores):
    avg_scores = (yamnet_scores + wav2vec2_scores) / 2
    return avg_scores.flatten()

@activity.defn
async def receive_audio(input_data):
    """Receive the audio stream or file and preprocess it into chunks."""
    file_name = input_data["file_name"]
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    return wav_data, sr

@activity.defn
async def process_audio(waveform, sr):
    """Process the audio chunk and get the classification predictions."""
    waveform = waveform / 32768.0
    waveform = waveform.astype('float32')
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)
    
    max_length = 160000
    if waveform.shape[0] < max_length:
        waveform = np.pad(waveform, (0, max_length - waveform.shape[0]))
    else:
        waveform = waveform[:max_length]

    inputs_wav2vec2 = processor_wav2vec2(waveform, return_tensors="pt", sampling_rate=params.sample_rate)
    inputs_wav2vec2 = {key: val.squeeze(0).unsqueeze(0).to(device) for key, val in inputs_wav2vec2.items()}
    
    yamnet_scores, _, _ = yamnet(waveform)
    yamnet_scores = np.mean(yamnet_scores, axis=0)

    wav2vec2_outputs = wav2vec2_model(**inputs_wav2vec2)
    wav2vec2_scores = torch.nn.functional.softmax(wav2vec2_outputs.logits, dim=-1).cpu().detach().numpy()

    final_predictions = ensemble_average(yamnet_scores, wav2vec2_scores)
    top5_i = np.argsort(final_predictions)[::-1][:5]
    top5_probs = final_predictions[top5_i]

    return top5_i, top5_probs

@activity.defn
async def store_results(predictions):
    """Store or manage the classification results."""
    top5_i, top5_probs = predictions
    results = []
    for i in range(5):
        results.append(f'{yamnet_classes[top5_i[i]]}: {top5_probs[i]:.3f}')
    return results

@workflow.defn
class AudioClassificationWorkflow:
    @workflow.run
    async def run(self, input_data):
        audio_data = await workflow.execute_activity(receive_audio, input_data)
        waveform, sr = audio_data
        predictions = await workflow.execute_activity(process_audio, waveform, sr)
        results = await workflow.execute_activity(store_results, predictions)
        return results

async def main():
    # Connect to the Temporal server (awaited connection)
    client = await Client.connect("localhost:7233")  # await the connection

    # Set up the Temporal worker
    worker = Worker(client, task_queue="audio-classification-queue")
    
    # Register activities and workflow
    worker.register_workflow_implementation_type(AudioClassificationWorkflow)
    worker.register_activity_implementation_type(receive_audio)
    worker.register_activity_implementation_type(process_audio)
    worker.register_activity_implementation_type(store_results)
    
    # Run the worker
    await worker.run()

# Run the main function using asyncio
asyncio.run(main())
