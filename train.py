from datetime import datetime

import numpy as np
import pandas as pd


def extract_data():

    cgm_df = pd.read_csv("CGMData.csv")

    insulin_df = pd.read_csv("InsulinData.csv")

    return cgm_df, insulin_df


def parse_datetime(cgm_df, insulin_df):
    cgm_date = cgm_df["Date"]
    cgm_time = cgm_df["Time"]
    cgm_datetime = [
        datetime.strptime(cgm_date[i] + " " + cgm_time[i], "%m/%d/%Y %H:%M:%S")
        for i in range(len(cgm_date))
    ]

    cgm_df["Datetime"] = cgm_datetime  # Add the datetime to the DataFrame

    insulin_date = insulin_df["Date"]
    insulin_time = insulin_df["Time"]
    insulin_datetime = [
        datetime.strptime(insulin_date[i] + " " + insulin_time[i], "%m/%d/%Y %H:%M:%S")
        for i in range(len(insulin_date))
    ]

    insulin_df["Datetime"] = insulin_datetime  # Add the datetime to the DataFrame

    return cgm_datetime, insulin_datetime


def extract_meal_data(cgm_df, insulin_df, cgm_datetime, insulin_datetime):
    cgm = cgm_df["Sensor Glucose (mg/dL)"]
    meals_start_times = insulin_df[
        (insulin_df["BWZ Carb Input (grams)"].isna() == False)
        & (insulin_df["BWZ Carb Input (grams)"])
        != 0
    ]["Datetime"]
    meal_data = dict()
    valid_meals = list()
    print(len(meals_start_times))
    for i, datetime in enumerate(meals_start_times):
        if i == 0:
            valid_meals.append(datetime)
            continue
        if valid_meals[-1] < datetime + pd.Timedelta(hours=2):
            continue
        valid_meals.append(datetime)

    print(len(np.asarray(valid_meals)))
    return meals_start_times


def main():
    cgm_df, insulin_df = extract_data()
    cgm_datetime, insulin_datetime = parse_datetime(cgm_df, insulin_df)
    meals_start_times = extract_meal_data(
        cgm_df, insulin_df, cgm_datetime, insulin_datetime
    )


main()
