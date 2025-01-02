# AUDIO CLASSIFIER FOR CAR AND TRAM AUDIO SAMPLES

## Overview
This project implements a machine learning based audio classifier for car and tram sounds. The classifier classifies a new sample to either car class or tram class.
This is a 2 persons group project from the introduction to audio processing course in December 2024. For this project we downloaded the training data (2000+ samples) online and manually recorded car and tram audio samples (40+ samples) with mobile phones for the test and validation data. Because of the course instructions, the testing and validation have extremely few samples, thus giving unrealistically positive results for the classifier (100% accuracy).

## How to run?
1. Clone the project.
2. Install jupyter notebook and open audioClassifier.ipynb in a code editor.
3. Install the external libraries used in the code. Next to import statements there are comments stating the used version numbers, that have worked. Especially different Numpy versions may cause errors.
4. You may run all code blocks in order.

## What did I do in this project?
- The overall structure and design of the project.
- Feature extraction.
- Choosing and using the machine learning model.
- Built the overall pipeline.

Basically everything in the code, but the individual functions for loading and normalizing audio samples.

## Additional instructions
- You may add additional audio samples to their respective folders.
- You may use the already saved ML model or fit your own using different data.
