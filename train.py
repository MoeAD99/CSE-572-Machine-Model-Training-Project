from datetime import datetime as dt
from pprint import pprint

import numpy as np
import pandas as pd


def flatten(seq, container=None):
    if container is None:
        container = []

    for s in seq:
        try:
            iter(s)  # check if it's iterable
        except TypeError:
            container.append(s)
        else:
            flatten(s, container)

    return container


def extract_data():
    cgm_df = pd.read_csv("CGMData.csv")
    insulin_df = pd.read_csv("InsulinData.csv")
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


def extract_cgm_data(cgm_df, insulin_df, cgm_datetime, insulin_datetime):
    meals_start_times = insulin_df[
        (insulin_df["BWZ Carb Input (grams)"].isna() == False)
        & (insulin_df["BWZ Carb Input (grams)"])
        != 0
    ]["Datetime"]

    print("total meal start times: ", len(meals_start_times))

    meal_period = []
    for time in meals_start_times:
        meal_period.append(
            insulin_df[
                (insulin_df["Datetime"] >= time)
                & (insulin_df["Datetime"] <= time + pd.Timedelta(hours=2))
            ]["Datetime"]
        )
    # meal_period = [j for i in meal_period for j in i]
    # print(type(meal_period[0]))
    # meal_period = np.asarray(meal_period).flatten()
    # Initialize an empty list to store the start times of postprandial periods
    postprandial_period_start_times = []
    # Iterate over each valid meal start time
    for datetime in meals_start_times:
        # Check if the current datetime exists in the cgm_datetime list
        if datetime in cgm_datetime:
            # If it does, append it to the postprandial_period_start_times list
            postprandial_period_start_times.append(datetime)
        else:
            # If it doesn't, append the previous datetime (time right after the meal was taken) from the cgm_datetime list
            cgm_time_after_meal_intake = cgm_df[cgm_df["Datetime"] > datetime][
                "Datetime"
            ].min()
            if cgm_time_after_meal_intake - datetime > pd.Timedelta(hours=2):
                continue

            postprandial_period_start_times.append(cgm_time_after_meal_intake)
            # index = cgm_df[
            #     (cgm_df["Datetime"] >= cgm_time_after_meal_intake)
            #     & (
            #         cgm_df["Datetime"]
            #         < cgm_time_after_meal_intake + pd.Timedelta(hours=2)
            #     )
            # ].index
            # cgm_df["Meal Period"].iloc[index] = "true"
    print("total postprandial start times: ", len(postprandial_period_start_times))
    postprandial_datetimes = dict()
    for start_time in postprandial_period_start_times:
        postprandial_datetimes[start_time] = cgm_df[
            (cgm_df["Datetime"] >= start_time)
            & (cgm_df["Datetime"] <= start_time + pd.Timedelta(hours=2))
        ]["Datetime"]

    print("total postprandial datetimes: ", len(postprandial_datetimes.keys()))

    valid_meals_start_times = []

    # Iterate over the meals start times
    for i, datetime in enumerate(postprandial_period_start_times):
        # Append the last meal start time (since the dates in the DataFrame are in descending order) to the valid_meals_start_times list
        if i == 0:
            valid_meals_start_times.append(datetime)
            continue
        # If the last valid meal start time is in between the current meal start time plus 2 hours ahead
        if valid_meals_start_times[-1] < datetime + pd.Timedelta(hours=2):
            # Skip the current meal start time
            continue
        # Otherwise, append the current meal start time to the valid_meals_start_times list
        valid_meals_start_times.append(datetime)

    valid_postprandial_datetimes = []
    for start_time in valid_meals_start_times:
        valid_postprandial_datetimes.append(
            cgm_df[
                (cgm_df["Datetime"] >= start_time)
                & (cgm_df["Datetime"] <= start_time + pd.Timedelta(hours=2))
            ]["Datetime"]
        )
    print("valid_postprandial_datetimes: ", len(flatten(valid_postprandial_datetimes)))

    # print(np.asarray(valid_meals_start_times)[:20])
    # print(np.asarray(postprandial_period_start_times)[:20])
    meal_stretch_cgm_data = []
    meal_stretch_datetimes = dict()
    postprandial_period_datetimes = []
    postabsorptive = []
    # Iterate over each postprandial period start time
    for i, start_time in enumerate(valid_meals_start_times):
        # Extract the sensor glucose values from the CGM data
        # for the time period between 0.5 hours before the start time
        # and 2 hours after the start time
        # if i == 0:
        #     postabsorptive.append(
        #         cgm_df[cgm_df["Datetime"] > start_time + pd.Timedelta(hours=2)][
        #             "Datetime"
        #         ].iloc[-1]
        #     )
        # else:
        #     postabsorptive.append(
        #         cgm_df[
        #             (cgm_df["Datetime"] > start_time + pd.Timedelta(hours=2))
        #             & (cgm_df["Datetime"] < postprandial_period_start_times[i - 1])
        #         ]["Datetime"].iloc[-1]
        #     )
        stretch_datetimes = (
            cgm_df["Datetime"] < start_time + pd.Timedelta(hours=2)
        ) & (cgm_df["Datetime"] >= start_time - pd.Timedelta(hours=0.5))

        meal_stretch_cgm_data.append(
            cgm_df[stretch_datetimes]["Sensor Glucose (mg/dL)"]
        )

        meal_stretch_datetimes[start_time] = np.asarray(
            cgm_df[stretch_datetimes]["Datetime"]
        )

        postprandial_period_datetimes.append(
            cgm_df[
                (cgm_df["Datetime"] <= start_time + pd.Timedelta(hours=2))
                & (cgm_df["Datetime"] >= start_time)
            ]["Datetime"]
        )
    # for i in meal_stretch_cgm_data:
    #     print(len(i))

    # postprandial_period_datetimes = [j for i in postprandial_period_datetimes for j in i]
    for key in meal_stretch_datetimes.keys():
        meal_stretch_datetimes[key] = np.array(
            [pd.Timestamp(i) for i in meal_stretch_datetimes[key]]
        )

    no_meal_start_times = []

    for time in valid_meals_start_times:
        if (time + pd.Timedelta(hours=2)) not in postprandial_period_start_times:
            no_meal_start_times.append(
                insulin_df[insulin_df["Datetime"] > time + pd.Timedelta(hours=2)][
                    "Datetime"
                ].min()
            )
    print("no_meal_start_times: ", len(no_meal_start_times))
    # for i, start_time in enumerate(no_meal_start_times):
    #     if start_time + pd.Timedelta(hours=2):

    postabsorptive_period_start_times = []
    for datetime in no_meal_start_times:
        if datetime in cgm_datetime:
            postabsorptive_period_start_times.append(datetime)
        else:
            cgm_time_after_postprandial_end = cgm_df[cgm_df["Datetime"] > datetime][
                "Datetime"
            ].min()
            postabsorptive_period_start_times.append(cgm_time_after_postprandial_end)
    print("postabsorptive_period_start_times: ", len(postabsorptive_period_start_times))
    total_postabsorptive_datetimes = []
    for i in range(len(postprandial_period_start_times)):
        meal_start = postprandial_period_start_times[i]
        meal_end = meal_start + pd.Timedelta(hours=2)
        # next_meal_start = postprandial_period_start_times[i + 1]
        # next_meal_end = next_meal_start + pd.Timedelta(hours=2)
        if i == 0:
            total_postabsorptive_datetimes.append(
                cgm_df[(cgm_df["Datetime"] > meal_end)]["Datetime"]
            )
        elif i == len(postprandial_period_start_times):
            total_postabsorptive_datetimes.append(
                cgm_df[(cgm_df["Datetime"] < meal_start)]["Datetime"]
            )
        else:
            previous_meal_start = postprandial_period_start_times[i - 1]
            if previous_meal_start - meal_end > pd.Timedelta(hours=2):
                total_postabsorptive_datetimes.append(
                    cgm_df[
                        (cgm_df["Datetime"] < previous_meal_start)
                        & (cgm_df["Datetime"] > meal_end)
                    ]["Datetime"]
                )
        # if i == 0:
        #     postabsorptive_datetimes.append(
        #         cgm_df[
        #             (cgm_df["Datetime"] > meal_end)
        #             | (
        #                 (cgm_df["Datetime"] < meal_start)
        #                 & (cgm_df["Datetime"] > next_meal_end)
        #             )
        #         ]["Datetime"]
        #     )
        # else:
        #     previous_meal_start = postprandial_period_start_times[i - 1]
        #     postabsorptive_datetimes.append(
        #         cgm_df[
        #             (
        #                 (cgm_df["Datetime"] < previous_meal_start)
        #                 & (cgm_df["Datetime"] > meal_end)
        #             )
        #             | (
        #                 (cgm_df["Datetime"] < meal_start)
        #                 & (cgm_df["Datetime"] > next_meal_end)
        #             )
        #         ]["Datetime"]
        #     )

    sum = 0
    for key, val in postprandial_datetimes.items():
        sum += len(list(val))
    print("postprandial_datetimes", sum)
    print(
        "total_postabsorptive_datetimes: ", len(flatten(total_postabsorptive_datetimes))
    )
    # total_postabsorptive_datetimes = np.array(
    #     [i.to_numpy() for i in total_postabsorptive_datetimes], dtype=dt
    # )
    total_postabsorptive_datetimes = [
        j for i in total_postabsorptive_datetimes for j in i
    ]
    # print(type(np.array(total_postabsorptive_datetimes, dtype=dt)[0]))
    # print(len(total_postabsorptive_datetimes))
    no_meal_stretch_datetimes = []
    i = 0
    j = 0
    total_postabsorptive_start_times = []
    no_meal_stretch_cgm_data = []

    for start_time in postabsorptive_period_start_times[::-1]:
        i += 1
        total_postabsorptive_start_times.append(start_time)
        interval_end = start_time + pd.Timedelta(hours=2)
        cgm_stretch = cgm_df[
            (cgm_df["Datetime"] >= start_time) & (cgm_df["Datetime"] < interval_end)
        ]["Sensor Glucose (mg/dL)"]
        no_meal_stretch_cgm_data.append(cgm_stretch)
        for dt in total_postabsorptive_datetimes[::-1]:
            j += 1
            if start_time <= dt <= interval_end:
                no_meal_stretch_datetimes.append(dt)
            if dt >= interval_end:
                if dt + pd.Timedelta(hours=2) in total_postabsorptive_datetimes:
                    start_time = dt
                    interval_end = start_time + pd.Timedelta(hours=2)
                    cgm_stretch = cgm_df[
                        (cgm_df["Datetime"] >= start_time)
                        & (cgm_df["Datetime"] < interval_end)
                    ]["Sensor Glucose (mg/dL)"]
                    no_meal_stretch_cgm_data.append(cgm_stretch)
                    total_postabsorptive_start_times.append(start_time)
                else:
                    break

    print(len(total_postabsorptive_start_times))
    print(len(meal_stretch_cgm_data))
    print(len(no_meal_stretch_cgm_data))
    meal_stretch_cgm_data = [
        x for x in meal_stretch_cgm_data if len(x) == 30 and not x.hasnans
    ]
    no_meal_stretch_cgm_data = [
        x.fillna(0).interpolate(method="polynomial", order=2)
        for x in no_meal_stretch_cgm_data
        if len(x) == 24 and x.isna().sum() <= 4
    ]
    print(np.sum([len(i) != 30 for i in meal_stretch_cgm_data]))
    print(np.sum([len(i) != 24 for i in no_meal_stretch_cgm_data]))
    # print(meal_stretch_cgm_data)
    print(np.array(meal_stretch_cgm_data).shape)
    print(np.array(no_meal_stretch_cgm_data).shape)

    return np.array(meal_stretch_cgm_data[::-1]), np.array(
        no_meal_stretch_cgm_data[::-1]
    )


def main():
    cgm_df, insulin_df = extract_data()
    cgm_datetime, insulin_datetime = parse_datetime(cgm_df, insulin_df)
    meal_cgm_data, no_meal_cgm_data = extract_cgm_data(
        cgm_df, insulin_df, cgm_datetime, insulin_datetime
    )
    print("min mean meal    ", meal_cgm_data.min(axis=1).mean())
    print("min mean no meal ", no_meal_cgm_data.min(axis=1).mean())

    print("max mean meal    ", meal_cgm_data.max(axis=1).mean())
    print("max mean no meal ", no_meal_cgm_data.max(axis=1).mean())


main()
