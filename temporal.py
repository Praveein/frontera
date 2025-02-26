import numpy as np
import torch
import librosa
import tensorflow as tf
import soundfile as sf
import resampy  # Import resampy
import asyncio
import tensorflow_hub as hub
from collections import namedtuple
from temporalio import workflow, activity
from temporalio.worker import Worker
from temporalio.client import Client
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import params as yamnet_params
import yamnet as yamnet_model

# Define a named tuple for structured result storage
AudioClassificationResult = namedtuple('AudioClassificationResult', ['audio_file', 'prediction', 'confidence'])

# Load YAMNet model from TensorFlow Hub
yamnet_model_url = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_url)


# Load Wav2Vec2 model from Huggingface's transformers
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=521)
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Initialize the Params class to access model parameters
params = yamnet_params.Params()

# Load YAMNet model and processor
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

# Specify the directory where the model and config are saved
model_path = r"results\checkpoint-1"  # Replace with your actual path

# Use the processor from the base model as a fallback (if preprocessor_config.json is missing)
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

processor_wav2vec2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec2_model.to(device)

# Ensemble method: Averaging Probabilities from both models
def ensemble_average(yamnet_scores, wav2vec2_scores):
    """Combine predictions from YAMNet and Wav2Vec2 by averaging probabilities."""
    avg_scores = (yamnet_scores + wav2vec2_scores) / 2
    return avg_scores.flatten()  # Return the averaged scores (not the indices)

# Activity functions in Temporal
@activity.defn
async def preprocess_audio(file_name: str, params: yamnet_params.Params) -> np.ndarray:
    """Preprocess audio for prediction."""
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

    return waveform

@activity.defn
async def run_ensemble_model(audio_chunk: np.ndarray, yamnet, wav2vec2_model, processor_wav2vec2, params: yamnet_params.Params) -> int:
    """Run the ensemble model to predict the class."""
    # Preprocess audio for Wav2Vec2 (use Wav2Vec2Processor)
    inputs_wav2vec2 = processor_wav2vec2(audio_chunk, return_tensors="pt", sampling_rate=params.sample_rate)

    # Ensure the input tensor has a batch dimension of 1 (for a batch of size 1)
    inputs_wav2vec2 = {key: val.squeeze(0).unsqueeze(0).to(device) for key, val in inputs_wav2vec2.items()}  # Add batch dimension correctly

    # Get YAMNet predictions (scores)
    yamnet_scores, _, _ = yamnet(audio_chunk)
    yamnet_scores = np.mean(yamnet_scores, axis=0)  # Average the scores over time

    # Get Wav2Vec2 predictions (logits)
    wav2vec2_outputs = wav2vec2_model(**inputs_wav2vec2)
    wav2vec2_scores = torch.nn.functional.softmax(wav2vec2_outputs.logits, dim=-1).cpu().detach().numpy()

    # Combine the predictions using ensemble averaging
    final_predictions = ensemble_average(yamnet_scores, wav2vec2_scores)

    # Get the top 5 predictions and their probabilities
    top5_i = np.argsort(final_predictions)[::-1][:5]  # Sort indices in descending order to get top 5
    top5_probs = final_predictions[top5_i]  # Get the probabilities of the top 5 predictions

    return top5_i, top5_probs

@activity.defn
async def store_result(result: AudioClassificationResult):
    """Store the result in a database or file."""
    print(f"Storing result for {result.audio_file}: Predicted class {result.prediction} with confidence {result.confidence}")

# Temporal Workflow
@workflow.defn
class AudioClassificationWorkflow:
    def __init__(self, ensemble_model: EnsembleModel):
        self.ensemble_model = ensemble_model

    @workflow.run
    async def classify_audio(self, audio_file: str, audio_chunk: np.ndarray) -> AudioClassificationResult:
        """Run the entire audio classification workflow."""
        
        # Step 1: Preprocess the audio (if needed)
        preprocessed_audio = await workflow.execute_activity(preprocess_audio, audio_file, params)
        
        # Step 2: Run ensemble model prediction
        prediction, top5_probs = await workflow.execute_activity(run_ensemble_model, preprocessed_audio, yamnet, wav2vec2_model, processor_wav2vec2, params)
        
        # Step 3: Store or return the result
        result = AudioClassificationResult(audio_file, prediction[0], confidence=top5_probs[0])  # Assuming high confidence
        await workflow.execute_activity(store_result, result)
        
        return result

# Worker code to run the workflow
async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(client, task_queue="audio-classification-task-queue")

    # Register the workflow with Temporal
    worker.register_workflow_implementation_type(AudioClassificationWorkflow)

    # Start the worker
    await worker.run()

# Start the Temporal worker
if __name__ == "__main__":
    asyncio.run(main())
