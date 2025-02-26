import sys
import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from tensorflow.keras import layers, Model
from yamnet import yamnet_frames_model
from params import Params


YAMNET_PATH = "yamnet.h5"


def create_dataset(path):
    samples, labels = [], []
    model = yamnet_frames_model(Params())
    model.load_weights(YAMNET_PATH)

    # Iterate over class directories
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)

        # Check if it's a directory (class folder)
        if os.path.isdir(cls_path):
            audio_files = os.listdir(cls_path)  # List audio files in the class folder

            print(f"Processing class: {cls} with {len(audio_files)} files")  # Debugging
            
            for sound in tqdm(audio_files, desc=f"Processing class {cls}"):
                # Construct the file path and replace backslashes with forward slashes
                sound_path = os.path.join(cls_path, sound).replace("\\", "/")  # Replace backslashes with forward slashes
                
                # Debugging: Check the file path
                print(f"Loading file: {sound_path}")
                
                # Check if the file is a valid audio file
                if sound_path.endswith(('.mp3', '.wav')):
                    try:
                        # Load the audio file at 16 kHz
                        wav = librosa.load(sound_path, sr=16000)[0].astype(np.float32)

                        # Here you can add preprocessing, augmentations, silence removal, etc.
                        for feature in model(wav)[1]:
                            samples.append(feature)
                            labels.append(cls)
                    except Exception as e:
                        print(f"Error loading {sound_path}: {e}")
                else:
                    print(f"Skipping non-audio file: {sound_path}")

    print(f"Total samples: {len(samples)}, Total labels: {len(labels)}")  # Debugging
    
    samples = np.asarray(samples)
    labels = np.asarray(labels)
    
    if len(labels) == 0:
        print("Warning: No valid audio files found or loaded.")
    
    return samples, labels



def generate_model(num_classes,
                  num_hidden=64,
                  activation='softmax',
                  regularization=0.03,
                  ):

    input = layers.Input(shape=(1024,))
    net = layers.Dense(num_hidden, activation=None, kernel_regularizer=tf.keras.regularizers.l2(regularization))(input)
    net = layers.Dense(num_classes, activation=activation)(net)
    model = Model(inputs=input, outputs=net)
    return model


def train_model(X,
                y,
                fname,  # Path where to save the model
                activation='softmax',
                epochs=30,
                optimizer='adam',
                num_hidden=64,
                batch_size=64
                ):
    # Encode the labels
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(y)

    # Save the names of the classes for future using.
    np.save(fname, encoder.classes_)
    num_classes = len(np.unique(y))

    # Generate the model
    general_model = generate_model(num_classes, num_hidden=num_hidden, activation=activation)
    general_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    # Create some callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=15, verbose=1,
                                                      min_lr=0.000001)]

    general_model.fit(X, labels, epochs=epochs, validation_split=0.20, batch_size=batch_size,
                      callbacks=callbacks, verbose=1)

    # Load the best weights after the training.
    general_model.load_weights(fname)

    return general_model


def main(argv):
    assert argv, 'Usage: train.py <path_to_data> <model_name>'
    path = argv[0]
    fname = argv[1]
    samples, labels = create_dataset(path)
    model = train_model(samples, labels, fname)


if __name__ == "__main__":
    main(sys.argv[1:])
