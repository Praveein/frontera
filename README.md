# frontera
Audio Classifier with Yamnet and Wav2vec
The model was trained on a 521-label data set from https://research.google.com/audioset/ontology/index.html. This is due to the ease of availability of the Yamnet pre-trained model.
The model weights are uploaded at https://drive.google.com/drive/folders/1ZsDfeeMrYpKi08-4MCR3q3HLXXGpCgn_?usp=drive_link 
Ensemble code is in the Jupiter notebook
Temporal Docker code is at temporal.py whose config files are in their respective folder

Data Preparation
Audio from the dataset is downloaded and converted to .wav format approx (24gb) (2000+ data points)

Data Preprocessoring
For wav2vec: 
Step 1: Prepare Your Data
Organize your audio files in a directory structure, with subfolders for each category (cry, scream, and normal).
Make sure to include the yamnet_class_map.csv and ontology.json files.
Step 2: Train the Model
To train the model, simply run the following script:

bash
Copy
python train_model.py
This will start the training process and save the model after each epoch in the ./results directory.

Step 3: Evaluate the Model
After training, you can evaluate the model using the test set by running the same script. The evaluation results will be printed after each epoch.

Step 4: Save and Load the Model
The model is saved after each epoch in a checkpoint directory (e.g., ./results/checkpoint-{epoch}). To load the model later:

Step 5: Run Inference
You can run inference on a single audio file by loading the model and passing the audio data to the model:
Notes
The training loop handles batches of audio, skipping those with invalid labels (-1).
Audio files are padded or truncated to a fixed length for consistency.
The model uses the GPU if available, otherwise defaults to the CPU.

for Yamnet
YAMNet  requires downloading the following data file:

YAMNet model weights in Keras saved weights in HDF5 format.
After downloading this file into the same directory as this README, the installation can be tested by running python yamnet_test.py which runs some synthetic signals through the model and checks the outputs.

About the Model
The YAMNet code layout is as follows:

yamnet.py: Model definition in Keras.
params.py: Hyperparameters. You can usefully modify PATCH_HOP_SECONDS.
features.py: Audio feature extraction helpers.
inference.py: Example code to classify input wav files.
yamnet_test.py: Simple test of YAMNet installation

Ensemble model: 

You can install the necessary dependencies via pip:

bash
Copy
pip install torch tensorflow transformers librosa soundfile resampy numpy
Additional Dependencies
YAMNet model: Ensure you have the pre-trained YAMNet model weights available in the yamnet.h5 file.
YAMNet class map: Download the yamnet_class_map.csv for the class labels.
Dataset
The input data for this model consists of audio files (e.g., WAV files). In this project, the following steps are carried out for each input audio file:

Audio Preprocessing:

The audio is loaded and normalized.
If necessary, it is resampled to the required sample rate.
The waveform is trimmed or padded to a fixed size.
Model Input:

The processed audio is used as input for both the YAMNet and Wav2Vec2 models.
Key Steps in the Code
1. Model Loading
YAMNet: The YAMNet model is loaded with pre-trained weights (yamnet.h5) and the class labels from yamnet_class_map.csv.
Wav2Vec2: The Wav2Vec2 model is loaded from a saved checkpoint.
2. Ensemble Method
The predictions from both YAMNet and Wav2Vec2 are combined by averaging the probability scores from both models. This method helps to improve the robustness of the final classification.

3. Audio Processing
The audio file is loaded and preprocessed:

Normalization: The waveform is normalized to ensure consistent amplitude.
Resampling: If the sample rate differs from the required one, it is resampled to 16kHz using resampy.
Padding/Trimming: The audio waveform is adjusted to a fixed length of 160,000 samples.
4. Model Inference
Both models generate probability scores for each class. These scores are averaged to produce a final prediction. The top 5 predicted classes are displayed along with their probabilities.

How to Use
Step 1: Prepare Your Audio File
Ensure that you have an audio file in .wav format. For example, audiosets/ontology/Aircraft_0.wav.

Step 2: Run the Code
You can run the code by executing the script process_audio.py (or similar) as follows:

python
Copy
file_name = 'audiosets/ontology/Aircraft_0.wav'  # Replace with your audio file path
process_audio(file_name, yamnet, wav2vec2_model, processor_wav2vec2, params)
This will load the audio, preprocess it, and run the inference using both models. The top 5 predictions and their probabilities will be printed.

Step 3: Interpreting Results
The output will display the top 5 predicted classes along with their associated probabilities:

yaml
Copy
Aircraft_0.wav predictions:
  Aircraft    : 0.843
  Dog         : 0.056
  Music       : 0.042
  Rain        : 0.030
  Speech      : 0.020
Example Output
yaml
Copy
Aircraft_0.wav predictions:
  Aircraft    : 0.843
  Dog         : 0.056
  Music       : 0.042
  Rain        : 0.030
  Speech      : 0.020
This indicates that the model predicts the sound as "Aircraft" with a probability of 0.843, followed by "Dog", "Music", etc.

Notes
The ensemble method averages the output probabilities from both YAMNet and Wav2Vec2, which can improve classification performance over using either model alone.
The maximum waveform length is set to 160,000 samples to standardize the input size for both models.
Ensure that both models and the class map are correctly loaded for accurate results.

For Temporal Deployment
You can install the necessary dependencies with pip:

bash
Copy
pip install torch tensorflow temporalio transformers librosa soundfile resampy numpy pandas
Setup
Temporal Server
Before running the workflow, ensure you have a Temporal Server running. If you don't have one set up, you can follow the Temporal Documentation for installation.

You can start a local Temporal server by running:

bash
Copy
docker-compose up
Model Files
You will need the following files to run the code:

YAMNet model weights (yamnet.h5): This file contains the pre-trained YAMNet model weights.
YAMNet class map (yamnet_class_map.csv): The class map for YAMNet, mapping class names to indices.
Wav2Vec2 checkpoint: This can be the trained model checkpoint (checkpoint-1) for the Wav2Vec2 model.
Make sure these files are available and their paths are correctly specified in the code.

Key Components in the Code
1. Workflow Definition (AudioClassificationWorkflow)
The main workflow, AudioClassificationWorkflow, orchestrates the audio classification process:

classify_audio: Classifies a single audio file.
classify_audio_stream: Classifies a real-time audio stream (processed in chunks).
2. Activities
The following activities are executed within the workflow:

preprocess_audio: Loads and preprocesses the audio file (e.g., normalization, resampling).
run_yamnet: Runs YAMNet on the preprocessed audio and returns the predicted scores.
run_wav2vec2: Runs Wav2Vec2 on the preprocessed audio and returns the predicted scores.
ensemble_average: Combines the predictions from YAMNet and Wav2Vec2 using ensemble averaging.
store_results: Stores the final classification results (this can be extended to save to a database or file).
3. Main Execution (main)
The main function initializes the Temporal worker, which polls for tasks and executes the defined activities. It connects to the Temporal server and registers the workflow and activities.

4. Starting a Workflow (start_workflow)
This function allows you to start a workflow to classify a specific audio file. The workflow will execute the classification process asynchronously.

How to Use
Step 1: Start the Temporal Worker
To start the Temporal worker, run the following:

bash
Copy
python run_temporal_worker.py
This will start the worker that listens for tasks on the audio-classification-queue.

Step 2: Start the Audio Classification Workflow
To start the workflow and classify an audio file, run the following:

python
Copy
asyncio.run(start_workflow("audiosets/ontology/Aircraft_0.wav"))
Make sure to replace "audiosets/ontology/Aircraft_0.wav" with the path to your own audio file.

Step 3: Workflow Execution
The workflow will execute the following tasks:

Preprocessing: The audio will be loaded, normalized, and resampled.
Inference: Both YAMNet and Wav2Vec2 will process the audio and return their predictions.
Ensemble: The predictions from both models will be averaged.
Results: The top 5 predictions will be returned along with their probabilities.
Example Output
When you run the workflow, the results will be printed out like this:

css
Copy
Workflow result: {
    "audio_file": "audiosets/ontology/Aircraft_0.wav",
    "predictions": [
        {"class": "Aircraft", "probability": 0.843},
        {"class": "Dog", "probability": 0.056},
        {"class": "Music", "probability": 0.042},
        {"class": "Rain", "probability": 0.030},
        {"class": "Speech", "probability": 0.020}
    ]
}
Step 4: Extend the Workflow for Streaming
If you want to classify real-time audio streams (e.g., in 5-15 second chunks), you can use the classify_audio_stream method, passing a list of audio chunks.

Notes
Make sure you have a working Temporal server running locally or on the cloud.
Ensure that all model files and dependencies are correctly set up and paths are configured properly in the code.
The worker and client interact asynchronously to allow for efficient and scalable processing of audio classification tasks.
