import librosa
import numpy as np

SAMPLINGRATE: int = 16000

def loadAllAudio(fileName: str):
    audios: list = []
    # TODO: Get labels
    labels: list = []
    # TODO: Add loop to load all files.
    audio: np.ndarray
    samplingFreq: int
    audio, samplingFreq = librosa.load(fileName, sr=SAMPLINGRATE) # This automatically scales to 16kHz

    audios.append(audio)
    return audios, labels

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
    # TODO: convert audio to wav?
    # TODO: normalize amplitude and maybe some other stuff: duration?
    audio, labels = loadAllAudio('Auto1.m4a', 16000)
    features = extract_features(audio[0])
    # TODO: Split data to train and test. Use random shuffle? Also then shuffle labels accordingly.
    # TODO: Train model





if __name__ == '__main__':
    main()