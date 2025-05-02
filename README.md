# A_HR_modality

## Overview
This repository contains all the code that:
1. Optimises and trains an LSTM model to estimate HR from a speech signal
2. Trains an encoder for 1st stage training in final implementation of project

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/projectSERN/audio_hr_v2.git
    cd audio_hr_v2
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Final model parameters
### A_HR estimator
Uses the `LSTMHiddenSummation` model found [here](src/models.py) with the following model parameters

| Parameter | Value |
|----------------|----------------|
| Input dimension  | 13           | 
| Hidden size      | 32           | 
| Number of layers | 5            |
| Dropout          | 0.4          |
| Output dimension | 1            |


### A_HR encoder
Uses the `AHR_ConvEncoder` model found [here](src/models.py) with the following model parameters

| Parameter | Value |
|----------------|----------------|
| Number of features| 1           | 
| Number of classes | 1           | 
| Kernel size       | 3           |
