import os
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

NORMALIZED_SAMPLINGRATE: int = 16000 # Hz
FIXED_DURATION: float = 5.0 # Seconds

def load_all_audio(folderPath: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    # List to store the audio data
    audios: list = []
    labels: list = []
    # Loop through the audio files in the given folder
    fileName: str
    for fileName in os.listdir(folderPath):
        # Path to the audio file
        filePath: str = os.path.join(folderPath, fileName)

        # Try to load the file, error just in case
        try:
            # Load the audio file
            audio: np.ndarray
            sr: int
            audio, sr = sf.read(filePath)

            # Resample if sampling rate is not 16000 as it should already be.
            if sr != NORMALIZED_SAMPLINGRATE:
                numSamples: int = int(len(audio) * NORMALIZED_SAMPLINGRATE / sr)
                audio = signal.resample(audio, numSamples)
                sr = NORMALIZED_SAMPLINGRATE

            # Append the data as a dictionary
            audios.append({
                "fileName": fileName,
                "audio": audio
            })
            labels.append(label)
        except Exception as e:
            print(f"Error loading {fileName}: {e}")

    return audios, labels


def normalize_audio(audio: np.ndarray) -> np.ndarray:

    # Frequency normalization. Removing irrelevant frequencies using bandpass filter.
    # Input the frequency range you want.

    # values must be 0 < x < 8000.
    audio = normalize_frequency(audio, 10, 7999)
    # Samples need to have a fixed duration for ML classifier model.
    audio = normalize_duration(audio, FIXED_DURATION, NORMALIZED_SAMPLINGRATE)

    # Amplitude normalization by Peak Normalization. Normalized -1 to 1.
    audio = normalize_amplitude(audio)
    return audio


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:

    # Take peak from every audio file and divide file by it.
    audioData: dict
    for audioData in audio:
        peak: int = np.max(np.abs(audioData["audio"]))

        audioData["audio"] = audioData["audio"] / peak

    return audio


def normalize_frequency(audio: np.ndarray, lowcut: int, highcut: int,
                         sr: int=NORMALIZED_SAMPLINGRATE) -> np.ndarray:

    # Normalize frequency range to nyquist
    nyquist: int = sr / 2
    low: int = lowcut / nyquist
    high: int = highcut / nyquist

    # Remove unwanted frequencies with a bandpass filter
    audioData: dict
    for audioData in audio:
        sos: np.ndarray = signal.butter(10, [low, high], btype='band', output='sos')
        audioData["audio"] = signal.sosfilt(sos, audioData["audio"])

    return audio

def normalize_duration(audio: np.ndarray, fixedDuration: float=FIXED_DURATION,
                        samplingRate: int=NORMALIZED_SAMPLINGRATE) -> np.ndarray:
    audioData: dict
    for audioData in audio:
        numSamples: int = int(fixedDuration * samplingRate)
        audioLength: int = len(audioData["audio"])
        if audioLength > numSamples:
            audioData["audio"] = audioData["audio"][:numSamples]
        else:
            padding = np.zeros(numSamples - len(audioData["audio"]))
            audioData["audio"] = np.concatenate([audioData["audio"], padding])
    return audio

def extract_features(audio: np.ndarray) -> np.ndarray:
    allFeatures: list = []
    audioData: dict
    for audioData in audio:
        audio = audioData["audio"]
        # Extract features
        mfccs: np.ndarray = librosa.feature.mfcc(y=audio, sr=NORMALIZED_SAMPLINGRATE, n_mfcc=13)
        delta: np.ndarray = librosa.feature.delta(mfccs)
        delta2: np.ndarray = librosa.feature.delta(mfccs, order=2)
        zcr: np.ndarray = librosa.feature.zero_crossing_rate(audio)
        # Should we take the mean of the features to lower the dimensionality?
        mfccsMean: np.ndarray = np.mean(mfccs.T, axis=0)
        deltaMean: np.ndarray = np.mean(delta.T, axis=0)
        delta2Mean: np.ndarray = np.mean(delta2.T, axis=0)
        zcrMean: np.ndarray = np.mean(zcr.T, axis=0)
        # Add concatenated features to the list of all features
        allFeatures.append(np.hstack([mfccsMean.T, deltaMean.T, delta2Mean.T, zcrMean.T]).flatten())

    sc: StandardScaler = StandardScaler()
    scaledFeatures: np.ndarray = sc.fit_transform(allFeatures)
    return np.array(scaledFeatures)

def main():
    # Load training data
    # For the sake of efficiency audio has been converted to .wav mono sound in advance.
    # This way it does not need to be done every time the program is tested and takes a lot less disk space.
    carAudio: np.ndarray
    carLabels: np.ndarray
    carAudio, carLabels = load_all_audio("carSamples", 0)

    tramAudio: np.ndarray
    tram_labels: np.ndarray
    tramAudio, tram_labels = load_all_audio("tramSamples", 1)

    # Normalize audio signals.
    carAudio: np.ndarray = normalize_audio(carAudio)
    tramAudio: np.ndarray = normalize_audio(tramAudio)
    audioSamples: np.ndarray = np.concatenate((carAudio, tramAudio), axis=0)
    yTrain: np.ndarray = np.concatenate((carLabels, tram_labels), axis=0)
    # Extract features from audio samples.
    xTrain: np.ndarray = extract_features(audioSamples)


    # Do the same steps for validation data

    # carAudioVal: np.ndarray
    # carLabelsVal: np.ndarray
    # carAudioVal, carLabelsVal = load_all_audio("carVal", 0)

    # tramAudioVal: np.ndarray
    # tram_labelsVal: np.ndarray
    # tramAudioVal, tram_labelsVal = load_all_audio("tramVal", 1)

    # carAudioVal: np.ndarray = normalize_audio(carAudioVal)
    # tramAudioVal: np.ndarray = normalize_audio(tramAudioVal)
    # audioSamplesVal: np.ndarray = np.concatenate((carAudioVal, tramAudioVal), axis=0)
    # yVal: np.ndarray = np.concatenate((carLabelsVal, tram_labelsVal), axis=0)
    # xVal: np.ndarray = extract_features(audioSamplesVal)


    # Do the same steps for test data
    carAudioTest: np.ndarray
    carLabelsTest: np.ndarray
    carAudioTest, carLabelsTest = load_all_audio("carTest", 0)

    tramAudioTest: np.ndarray
    tram_labelsTest: np.ndarray
    tramAudioTest, tram_labelsTest = load_all_audio("tramTest", 1)

    carAudioTest: np.ndarray = normalize_audio(carAudioTest)
    tramAudioTest: np.ndarray = normalize_audio(tramAudioTest)
    audioSamplesTest: np.ndarray = np.concatenate((carAudioTest, tramAudioTest), axis=0)
    yTest: np.ndarray = np.concatenate((carLabelsTest, tram_labelsTest), axis=0)
    xTest: np.ndarray = extract_features(audioSamplesTest)



    # Testing with larger amount of signals.
    # Gave interesting results, but it was said that our own data should preferably be validation and test data.
    # xValTest: np.ndarray
    # yValTest: np.ndarray
    # xTrain, xValTest, yTrain, yValTest = train_test_split(xTrain, yTrain, test_size=0.4, random_state=42)
    # Extract validation and test data. Validation data is used for tuning the model.
    # xVal, xTest, yVal, yTest = train_test_split(xValTest, yValTest, test_size=0.5, random_state=42)



    # Train the model
    clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100)
    clf.fit(xTrain, yTrain)

    # validate the model
    # yValPred: np.ndarray = clf.predict(xVal)
    # accuracy: float = accuracy_score(yVal, yValPred)
    # print(f"Validation Accuracy: {accuracy}")
    # print(classification_report(yVal, yValPred))

    # Test the model
    yTestPred: np.ndarray = clf.predict(xTest)
    accuracy: float = accuracy_score(yTest, yTestPred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(yTest, yTestPred, target_names=["Car", "Tram"]))

if __name__ == '__main__':
    main()