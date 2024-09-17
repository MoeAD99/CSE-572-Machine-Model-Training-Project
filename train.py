from datetime import datetime as dt

import numpy as np
import pandas as pd


def extract_data():

    cgm_df = pd.read_csv("CGMData.csv")
    cgm_df = cgm_df.drop_duplicates(keep="last", ignore_index=True)

    insulin_df = pd.read_csv("InsulinData.csv")
    insulin_df = insulin_df.drop_duplicates(keep="last", ignore_index=True)
    return cgm_df, insulin_df


def parse_datetime(cgm_df, insulin_df):
    cgm_date = cgm_df["Date"]
    cgm_time = cgm_df["Time"]
    cgm_datetime = [
        dt.strptime(cgm_date[i] + " " + cgm_time[i], "%m/%d/%Y %H:%M:%S")
        for i in range(len(cgm_date))
    ]

    cgm_df["Datetime"] = cgm_datetime  # Add the datetime to the DataFrame

    insulin_date = insulin_df["Date"]
    insulin_time = insulin_df["Time"]
    insulin_datetime = [
        dt.strptime(insulin_date[i] + " " + insulin_time[i], "%m/%d/%Y %H:%M:%S")
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

    meal_period = []
    for time in meals_start_times:
        meal_period.append(
            insulin_df[insulin_df["Datetime"] <= time + pd.Timedelta(hours=2)][
                "Datetime"
            ]
        )
    meal_period = list(np.asarray(meal_period).flatten())
    valid_meals_start_times = []
    print(len(meals_start_times))
    # Iterate over the meals start times
    for i, datetime in enumerate(meals_start_times):
        # Append the last meal start time (since the dates in the DataFrame are in descending order) to the valid_meals_start_times list
        if i == 0:
            valid_meals_start_times.append(datetime)
            continue
        # If the last valid meal start time is in between the current meal start time and 2 hours ahead
        if valid_meals_start_times[-1] < datetime + pd.Timedelta(hours=2):
            # Skip the current meal start time
            continue
        # Otherwise, append the current meal start time to the valid_meals_start_times list
        valid_meals_start_times.append(datetime)

    # Initialize an empty list to store the start times of postprandial periods
    postprandial_period_start_times = []
    # Iterate over each valid meal start time
    for datetime in valid_meals_start_times:
        # Check if the current datetime exists in the cgm_datetime list
        if datetime in cgm_datetime:
            # If it does, append it to the postprandial_period_start_times list
            postprandial_period_start_times.append(datetime)
        else:
            # If it doesn't, append the previous datetime (time right after the meal was taken) from the cgm_datetime list
            cgm_time_after_meal_intake = cgm_df[cgm_df["Datetime"] > datetime][
                "Datetime"
            ].iloc[-1]
            postprandial_period_start_times.append(cgm_time_after_meal_intake)

    # print(np.asarray(valid_meals_start_times)[:20])
    # print(np.asarray(postprandial_period_start_times)[:20])
    meal_stretch_cgm_data = dict()
    meal_stretch_datetimes = dict()
    postprandial_period_datetimes = []
    # Iterate over each postprandial period start time
    for start_time in postprandial_period_start_times:
        # Extract the sensor glucose values from the CGM data
        # for the time period between 0.5 hours before the start time
        # and 2 hours after the start time
        target_datetimes = (cgm_df["Datetime"] < start_time + pd.Timedelta(hours=2)) & (
            cgm_df["Datetime"] >= start_time - pd.Timedelta(hours=0.5)
        )

        meal_stretch_cgm_data[start_time] = np.asarray(
            cgm_df[target_datetimes]["Sensor Glucose (mg/dL)"]
        )

        meal_stretch_datetimes[start_time] = np.asarray(
            cgm_df[target_datetimes]["Datetime"]
        )

        postprandial_period_datetimes.append(
            cgm_df["Datetime"] < start_time + pd.Timedelta(hours=2)
        )

    for key in meal_stretch_datetimes.keys():
        meal_stretch_datetimes[key] = np.array(
            [pd.Timestamp(i) for i in meal_stretch_datetimes[key]]
        )

    no_meal_start_times = []

    for time in meals_start_times:
        if time + pd.Timedelta(hours=2) not in meal_period:
            no_meal_start_times.append(
                insulin_df[insulin_df["Datetime"] > time + pd.Timedelta(hours=2)][
                    "Datetime"
                ].iloc[-1]
            )
    # for i, start_time in enumerate(no_meal_start_times):
    #     if start_time + pd.Timedelta(hours=2):

    postabsorptive_period_start_times = []
    for datetime in no_meal_start_times:
        if datetime in cgm_datetime:
            postabsorptive_period_start_times.append(datetime)
        else:
            cgm_time_after_postprandial_end = cgm_df[cgm_df["Datetime"] > datetime][
                "Datetime"
            ].iloc[-1]
            postabsorptive_period_start_times.append(cgm_time_after_postprandial_end)

    total_postabsorptive_period_start_times = postabsorptive_period_start_times
    for time in postabsorptive_period_start_times:
        if time + pd.Timedelta(hours=2) not in postprandial_period_datetimes:
            total_postabsorptive_period_start_times.append(time + pd.Timedelta(hours=2))

    no_meal_stretch_cgm_data = dict()

    for time in total_postabsorptive_period_start_times:
        no_meal_stretch_cgm_data[time] = cgm_df[
            cgm_df["Datetime"] >= time + pd.Timedelta(hours=2)
        ]["Sensor Glucose (mg/dL)"]

    # print(
    #     pd.Timestamp(meal_stretch_datetimes[list(meal_stretch_datetimes.keys())[0]][0])
    # )
    print(np.asarray(no_meal_stretch_cgm_data).size)
    return meals_start_times


def main():
    cgm_df, insulin_df = extract_data()
    cgm_datetime, insulin_datetime = parse_datetime(cgm_df, insulin_df)
    meals_start_times = extract_meal_data(
        cgm_df, insulin_df, cgm_datetime, insulin_datetime
    )


main()
