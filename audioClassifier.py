import os
import librosa
import numpy as np
import sounddevice as sd
import scipy.signal as signal

def load_all_audio(folder_path: str):

    # List to store the audio data
    audios = []

    # Loop through the audio files in the given folder
    for file_name in os.listdir(folder_path):

        # Path to the audio file
        file_path = os.path.join(folder_path, file_name)
        print(f"Loading: {file_path}")

        # Try to load the file, error just in case
        try:
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=16000, mono=True)

            # Append the data as a dictionary
            audios.append({
                "fileName": file_name,
                "audio": audio
            })
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return audios


def normalize_audio(audio):

    # Frequency normalization. Removing irrelevant frequencies using bandpass filter.
    # Input the frequency range you want.

    # values must be 0 < x < 8000. Might want to mess with these when testing learning.
    audio = normalize_frequency(audio, 10, 7999)


    # Logarithmically scale audio

    # might be helpful or not
    #logarithmic_scaling(audio)


    # Amplitude normalization by Peak Normalization. Normalized -1 to 1.
    audio = normalize_amplitude(audio)

    return audio


def normalize_amplitude(audio):

    # Take peak from every audio file and divide file by it.
    for audio_data in audio:
        peak = np.max(np.abs(audio_data["audio"]))

        audio_data["audio"] = audio_data["audio"] / peak

    return audio


def normalize_frequency(audio, lowcut, highcut, sr=16000):

    # Normalize frequency range to nyquist
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Remove unwanted frequencies with a bandpass filter
    for audio_data in audio:
        sos = signal.butter(10, [low, high], btype='band', output='sos')
        audio_data["audio"] = signal.sosfilt(sos, audio_data["audio"])

    return audio

def logarithmic_scaling(audio, epsilon=1e-16):

    # Logarithmically scale audio. Add Epsilon in case of 0s.
    for audio_data in audio:
        audio_data["audio"] = np.sign(audio_data["audio"]) * np.log(np.abs(audio_data["audio"]) + epsilon)

    return audio

def extract_features(audio):
    mfccs: np.ndarray = librosa.feature.mfcc(y=audio, sr=SAMPLINGRATE, n_mfcc=13)
    delta: np.ndarray = librosa.feature.delta(mfccs)
    delta2: np.ndarray = librosa.feature.delta(mfccs, order=2)

    # Should we take the mean of the features to lower the dimensionality?
    mfccsMean: np.ndarray = np.mean(mfccs.T, axis=0)
    deltaMean: np.ndarray = np.mean(delta.T, axis=0)
    delta2Mean: np.ndarray = np.mean(delta2.T, axis=0)
    # TODO: Add 4th feature
    return [mfccsMean, deltaMean, delta2Mean]


def main():
    # Load audio data
    # For the sake of efficiency audio has been normalized to 16kHz mono sound in advance.
    # This way it does not need to be done every time the program is tested and takes a lot less disk space.
    car_audio = load_all_audio("carSamples")
    tram_audio = load_all_audio("tramSamples")

    # Normalize audio signals. For more information checkout the normalize_audio function.
    car_audio = normalize_audio(car_audio)
    tram_audio = normalize_audio(tram_audio)

    #features = extract_features(audio[0])
    # TODO: Split data to train and test. Use random shuffle? Also then shuffle labels accordingly.
    # TODO: Train model





if __name__ == '__main__':
    main()