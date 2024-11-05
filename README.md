# Efficient Data Stream Anomaly Detection

## Project Description:
This project involves developing a Python script for real-time anomaly detection in a continuous data stream. The stream simulates sequences of floating-point numbers that may represent different metrics, such as financial data or system performance metrics. The focus is on identifying unusual patterns, such as significantly high values or deviations from expected behavior

## Objectives:
1. Algorithm Selection: Implement an appropriate algorithm for anomaly detection that can adapt to changes in data trends and seasonal patterns.
2. Data Stream Simulation: Develop a function to simulate a data stream that includes regular patterns, seasonal trends, and random noise.
3. Anomaly Detection: Create a real-time system capable of detecting anomalies in the incoming data stream.
4. Optimization: Ensure the anomaly detection system is optimized for speed and performance.
Visualization: Implement a tool for real-time visualization of the data stream and any detected anomalies.

## Requirements:
joblib: 1.2.0
scikit-learn: 1.1.3
matplotlib: 3.9.2
numpy: 1.23.5

## Working
#### [main.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/main.py)
This is the main script that drives the project. It requires the following files:
1. [model_prod.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/model_prod.py)
2. [settings.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/settings.py)
3. [anomaly.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/anomaly.py)

#### [settings.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/settings.py)
This file contains parameters that control the behavior of the code. Modifying these parameters will change the output. It includes:

1. DELAY: The time interval between the generation of new data points.
2. OUTLIERS_GENERATION_PROBABILITY: The probability that a generated data point is an anomaly (e.g., if set to 0.2, there is a 20% chance).
3. VISUALIZATION: A flag to enable or disable real-time visualization.

#### [model_prod.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/model_prod.py)
This file defines and trains the Isolation Forest model used for anomaly detection. It generates synthetic data for training and fits the data to the Isolation Forest. The trained model is saved to isolation_forest.joblib.

#### [anomaly.py](https://github.com/ParagGawai/Efficient_Data_Stream_Anomaly_Detection/blob/main/anomaly.py)
This script defines the function anomaly_dect, which performs anomaly detection using the previously stored Isolation Forest model.

1. Configures logging to output information to anomaly.log.
2. Continuously generates random data points and evaluates them with the pre-trained Isolation Forest model.
3. Visualization is managed through the VISUALIZATION constant in settings.py.
4. Logs detected anomalies and marks them on the visualization if enabled.
5. Pauses for a specified delay (DELAY) between iterations for real-time visualization.
6. Runs indefinitely until the model file is unavailable or the loop is manually interrupted.
