import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from train import *


def transform_test_data(row):
    end_time = datetime(2024, 1, 1, 2, 0)
    times = [end_time - timedelta(minutes=5 * i) for i in range(24)]

    df = pd.DataFrame(
        {
            "Datetime": times,
            "Sensor Glucose (mg/dL)": row[::-1],
        }
    )
    return df


test_data = pd.read_csv("test.csv", header=None)

processed_samples = [transform_test_data(row) for _, row in test_data.iterrows()]

features = extract_no_meal_features(processed_samples)

with open("meal_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

pred_labels = model.predict(features)

pred_labels = pd.DataFrame(pred_labels)
pred_labels.to_csv("Result.csv", header=False, index=False)
