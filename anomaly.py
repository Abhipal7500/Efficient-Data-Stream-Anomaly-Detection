"""
This function, anomaly_dect, is designed to identify anomalies using a pre-trained Isolation Forest model.
"""

"""
1. The script configures logging to output information into a file called 'anomaly.log'.
2. The function anomaly_dect generates random data points in a loop and checks them against a pre-trained Isolation Forest model.
3. The option to visualize results is controlled by the VISUALIZATION constant defined in settings.py.
4. Detected anomalies are logged, and data points are highlighted in the visualization (if visualization is enabled).
5. The function waits for a set period (DELAY) between iterations to allow for visualization updates.
6. The loop runs indefinitely until a model file is missing or an interruption occurs.
"""

# Import required libraries
import os
import random
import time
from datetime import datetime
from joblib import load
import logging
import matplotlib.pyplot as plt
import numpy as np

# Import constants from the settings module
from settings import DELAY, OUTLIERS_GENERATION_PROBABILITY, VISUALIZATION

# Configure logging to output to 'anomaly.log'
logging.basicConfig(filename='anomaly.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# List to keep track of real-time data points
data_ls = []

def anomaly_dect():
    # Initialize an ID counter for incoming data points
    _id = 0

    # Set up visualization if enabled
    if VISUALIZATION:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_facecolor("#d3f9d8")  
        fig.show()

    while True:
        # Randomly generate anomalous data points with a certain probability
        if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
            X_test = np.random.uniform(low=-4, high=4, size=(1, 1))
        else:
            X = 0.3 * np.random.randn(1, 1)
            X_test = (X + np.random.choice(a=[2, -2], size=1, p=[0.5, 0.5]))

        X_test = np.round(X_test, 3).tolist()

        current_time = datetime.utcnow().isoformat()

        # Create a dictionary to store details of the incoming data
        record = {"id": _id, "data": X_test, "current_time": current_time}
        print(f"Incoming: {record}")

        # Attempt to load the Isolation Forest model from file
        try:
            model_path = os.path.abspath("isolation_forest.joblib")
            clf = load(model_path)
        except:
            logging.warning("Model file not found")
            print('Model file not available')
            break

        # Extract the data and add it to the list for visualization
        data = record['data']
        data_ls.append(data[0][0])
        prediction = clf.predict(data)

        # Update the plot if visualization is enabled
        if VISUALIZATION:
            ax.plot(data_ls, color='black')  
            fig.canvas.draw()
            ax.set_xlim(left=0, right=_id + 2)

        # Check if the prediction is an anomaly
        if prediction[0] == -1:
            score = clf.score_samples(data)
            record["score"] = np.round(score, 3).tolist()
            if VISUALIZATION:
                plt.scatter(_id, data_ls[_id], color='r', linewidth=2)
            logging.info(f"Anomaly Detected: {record}")
            print(f'Anomaly Detected: {record}')

        _id += 1
        plt.pause(DELAY)

    plt.show()
