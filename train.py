import pickle
from datetime import datetime as dt

import numpy as np
import pandas as pd
from scipy.fft import fft, rfft
from scipy.signal import find_peaks
from scipy.stats import variation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


def extract_data(cgm_file, insulin_file):
    cgm_df = pd.read_csv(cgm_file, low_memory=False)
    insulin_df = pd.read_csv(insulin_file, low_memory=False)
    return cgm_df, insulin_df


def parse_datetime(cgm_df, insulin_df):
    insulin_df["Datetime"] = pd.to_datetime(
        insulin_df["Date"] + " " + insulin_df["Time"]
    )
    cgm_df["Datetime"] = pd.to_datetime(cgm_df["Date"] + " " + cgm_df["Time"])


def extract_no_meal_data(
    valid_meals_start_times,
    postprandial_period_start_times,
    insulin_df,
    cgm_df,
):

    no_meal_start_times = []

    for time in valid_meals_start_times:
        if (time + pd.Timedelta(hours=2)) not in postprandial_period_start_times:
            no_meal_start_times.append(
                insulin_df[insulin_df["Datetime"] > time + pd.Timedelta(hours=2)][
                    "Datetime"
                ].min()
            )

    postabsorptive_period_start_times = []
    for datetime in no_meal_start_times:
        if datetime in cgm_df["Datetime"]:
            postabsorptive_period_start_times.append(datetime)
        else:
            cgm_time_after_postprandial_end = cgm_df[cgm_df["Datetime"] > datetime][
                "Datetime"
            ].min()
            postabsorptive_period_start_times.append(cgm_time_after_postprandial_end)
    total_postabsorptive_datetimes = []
    for i in range(len(postprandial_period_start_times)):
        meal_start = postprandial_period_start_times[i]
        meal_end = meal_start + pd.Timedelta(hours=2)
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

    total_postabsorptive_datetimes = [
        j for i in total_postabsorptive_datetimes for j in i
    ]

    no_meal_stretch_datetimes = []

    total_postabsorptive_start_times = []
    no_meal_stretch_cgm_data = []
    no_meal_stretch = []

    for start_time in postabsorptive_period_start_times[::-1]:
        total_postabsorptive_start_times.append(start_time)
        interval_end = start_time + pd.Timedelta(hours=2)
        stretch_datetimes = (cgm_df["Datetime"] >= start_time) & (
            cgm_df["Datetime"] < interval_end
        )
        cgm_stretch = cgm_df[stretch_datetimes]["Sensor Glucose (mg/dL)"]
        date_stretch = cgm_df[stretch_datetimes]["Datetime"]
        stretch = date_stretch = cgm_df[stretch_datetimes][
            ["Datetime", "Sensor Glucose (mg/dL)"]
        ]
        no_meal_stretch.append(stretch)
        no_meal_stretch_datetimes.append(date_stretch)
        no_meal_stretch_cgm_data.append(cgm_stretch)
        for dt in total_postabsorptive_datetimes[::-1]:
            # if start_time <= dt <= interval_end:
            #     no_meal_stretch_datetimes.append(dt)
            if dt >= interval_end:
                if dt + pd.Timedelta(hours=2) in total_postabsorptive_datetimes:
                    start_time = dt
                    interval_end = start_time + pd.Timedelta(hours=2)
                    stretch_datetimes = (cgm_df["Datetime"] >= start_time) & (
                        cgm_df["Datetime"] < interval_end
                    )
                    cgm_stretch = cgm_df[stretch_datetimes]["Sensor Glucose (mg/dL)"]
                    date_stretch = cgm_df[stretch_datetimes]["Datetime"]
                    stretch = cgm_df[stretch_datetimes][
                        ["Datetime", "Sensor Glucose (mg/dL)"]
                    ]
                    no_meal_stretch.append(stretch)
                    no_meal_stretch_datetimes.append(date_stretch)
                    no_meal_stretch_cgm_data.append(cgm_stretch)
                    total_postabsorptive_start_times.append(start_time)
                else:
                    break
    no_meal_stretch = [
        x
        for x in no_meal_stretch
        if len(x["Sensor Glucose (mg/dL)"]) == 24
        and not x["Sensor Glucose (mg/dL)"].hasnans
    ]

    no_meal_stretch_cgm_data = [
        x for x in no_meal_stretch_cgm_data if len(x) == 24 and not x.hasnans
    ]

    no_meal = [x["Sensor Glucose (mg/dL)"] for x in no_meal_stretch]
    return no_meal_stretch[::-1]


def extract_cgm_data(cgm_df, insulin_df):
    meals_start_times = insulin_df[
        (insulin_df["BWZ Carb Input (grams)"].isna() == False)
        & (insulin_df["BWZ Carb Input (grams)"])
        != 0
    ]["Datetime"]

    # print("total meal start times: ", len(meals_start_times))

    # Initialize an empty list to store the start times of postprandial periods
    postprandial_period_start_times = []
    # Iterate over each valid meal start time
    for datetime in meals_start_times:
        # Check if the current datetime exists in the cgm_datetime list
        if datetime in cgm_df["Datetime"]:
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

    # print("total postprandial start times: ", len(postprandial_period_start_times))
    postprandial_datetimes = dict()
    for start_time in postprandial_period_start_times:
        postprandial_datetimes[start_time] = cgm_df[
            (cgm_df["Datetime"] >= start_time)
            & (cgm_df["Datetime"] <= start_time + pd.Timedelta(hours=2))
        ]["Datetime"]

    # print("total postprandial datetimes: ", len(postprandial_datetimes.keys()))

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

    meal_stretch_cgm_data = []
    meal_stretch_datetimes = []
    meal_stretch = []
    postprandial_period_datetimes = []
    postabsorptive = []
    # Iterate over each postprandial period start time
    for i, start_time in enumerate(valid_meals_start_times):
        # Extract the sensor glucose values from the CGM data
        # for the time period between 0.5 hours before the start time
        # and 2 hours after the start time

        stretch_datetimes = (
            cgm_df["Datetime"] < start_time + pd.Timedelta(hours=2)
        ) & (cgm_df["Datetime"] >= start_time - pd.Timedelta(hours=0.5))

        meal_stretch_cgm_data.append(
            cgm_df[stretch_datetimes]["Sensor Glucose (mg/dL)"]
        )

        meal_stretch_datetimes.append(cgm_df[stretch_datetimes]["Datetime"])

        meal_stretch.append(
            cgm_df[stretch_datetimes][["Datetime", "Sensor Glucose (mg/dL)"]]
        )
        postprandial_period_datetimes.append(
            cgm_df[
                (cgm_df["Datetime"] <= start_time + pd.Timedelta(hours=2))
                & (cgm_df["Datetime"] >= start_time)
            ]["Datetime"]
        )

    meal_stretch = [
        x
        for x in meal_stretch
        if len(x["Sensor Glucose (mg/dL)"]) == 30
        and not x["Sensor Glucose (mg/dL)"].hasnans
    ]

    meal_stretch_cgm_data = [
        x for x in meal_stretch_cgm_data if len(x) == 30 and not x.hasnans
    ]

    meal = [x["Sensor Glucose (mg/dL)"] for x in meal_stretch]
    no_meal_stretch = extract_no_meal_data(
        valid_meals_start_times, postprandial_period_start_times, insulin_df, cgm_df
    )
    return meal_stretch, no_meal_stretch


def extract_meal_features(total_meal_data):
    total_meal_data = [x.reset_index() for x in total_meal_data]
    time_to_max_cgm_from_meal_start = []
    diff_cgm_max_meal = []
    features = []
    for meal_stretch in total_meal_data:

        meal_start_time = meal_stretch.iloc[23]["Datetime"]
        max_cgm_after_meal_index = meal_stretch.iloc[:23][
            "Sensor Glucose (mg/dL)"
        ].idxmax()
        max_cgm_time = meal_stretch.loc[max_cgm_after_meal_index]["Datetime"]
        time_diff = pd.Timedelta(max_cgm_time - meal_start_time).seconds
        time_to_max_cgm_from_meal_start.append(time_diff)

        max_cgm_after_meal = meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"].max()
        cgm_meal = meal_stretch.iloc[23]["Sensor Glucose (mg/dL)"]
        diff_cgm_max_meal.append((max_cgm_after_meal - cgm_meal) / cgm_meal)

        rfft_meal = list(abs(rfft(meal_stretch["Sensor Glucose (mg/dL)"].values)))
        rfft_meal_sorted = list(np.sort(rfft_meal))
        second_peak = rfft_meal_sorted[-2]
        second_peak_index = rfft_meal.index(second_peak)
        third_peak = rfft_meal_sorted[-3]
        third_peak_index = rfft_meal.index(third_peak)
        differential = np.mean(
            np.diff(list(meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"]))
        )
        second_differential = np.mean(
            np.diff(np.diff(list(meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])))
        )
        std = meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"].std()
        mean = meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"].mean()
        gradient = np.mean(
            np.gradient(meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])
        )
        variance = np.var(meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])
        var = variation(meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])
        min_cgm = meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"].min()
        features.append(
            [
                time_diff,
                (max_cgm_after_meal - cgm_meal) / cgm_meal,
                second_peak,
                second_peak_index,
                third_peak,
                third_peak_index,
                differential,
                second_differential,
            ]
        )
    return features


def extract_no_meal_features(total_no_meal_data):
    total_no_meal_data = [x.reset_index() for x in total_no_meal_data]
    time_to_max_cgm_from_no_meal_start = []
    diff_cgm_max_meal = []
    features = []
    for no_meal_stretch in total_no_meal_data:

        no_meal_start_time = no_meal_stretch.iloc[23]["Datetime"]
        max_cgm_index = no_meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"].idxmax()
        max_cgm_time = no_meal_stretch.loc[max_cgm_index]["Datetime"]
        time_diff = pd.Timedelta(max_cgm_time - no_meal_start_time).seconds
        time_to_max_cgm_from_no_meal_start.append(time_diff)

        max_cgm = no_meal_stretch["Sensor Glucose (mg/dL)"].max()
        initial_cgm = no_meal_stretch.iloc[23]["Sensor Glucose (mg/dL)"]
        diff = (max_cgm - initial_cgm) / initial_cgm
        diff_cgm_max_meal.append(diff)

        rfft_arr = list(abs(rfft(no_meal_stretch["Sensor Glucose (mg/dL)"].values)))
        rfft_sorted = list(np.sort(rfft_arr))
        second_peak = rfft_sorted[-2]
        second_peak_index = rfft_arr.index(second_peak)
        third_peak = rfft_sorted[-3]
        third_peak_index = rfft_arr.index(third_peak)
        differential = np.mean(np.diff(list(no_meal_stretch["Sensor Glucose (mg/dL)"])))
        second_differential = np.mean(
            np.diff(np.diff(list(no_meal_stretch["Sensor Glucose (mg/dL)"])))
        )

        std = no_meal_stretch["Sensor Glucose (mg/dL)"].std()
        mean = no_meal_stretch["Sensor Glucose (mg/dL)"].mean()
        gradient = np.mean(np.gradient(no_meal_stretch["Sensor Glucose (mg/dL)"]))
        variance = np.var(no_meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])
        var = variation(no_meal_stretch.iloc[:23]["Sensor Glucose (mg/dL)"])
        min_cgm = no_meal_stretch["Sensor Glucose (mg/dL)"].min()
        features.append(
            [
                time_diff,
                diff,
                second_peak,
                second_peak_index,
                third_peak,
                third_peak_index,
                differential,
                second_differential,
            ]
        )

    return features


def train(total_data_matrix):
    X = np.array([row[:-1] for row in total_data_matrix])  # Features
    y = np.array([row[-1] for row in total_data_matrix])  # Labels
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    scores = []
    predictions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = rf_model.predict(X_test)
        predictions.append(y_pred)
        # Calculate performance metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store the metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        scores.append(rf_model.score(X_test, y_test))
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Score: {np.mean(scores):.4f}")
    param_grid = {
        "n_estimators": [25, 50, 100, 150, 200, 250, 300],
        "random_state": [0, 42, 100, 101],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        verbose=3,
        scoring=["f1", "accuracy"],
        refit="f1",
        cv=10,
    )
    grid.fit(X_train, y_train)
    print("best hyperparameters: ", grid.best_params_)
    print(grid.best_estimator_)

    final_model = RandomForestClassifier(
        n_estimators=100, random_state=0, min_samples_split=5
    )
    # model = svm.SVC(kernel='rbf', probability=True,random_state=0)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("\nTest set results:")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    with open("meal_prediction_model.pkl", "wb") as file:
        pickle.dump(final_model, file)


def main():
    cgm_df, insulin_df = extract_data("CGMData.csv", "InsulinData.csv")
    cgm_df2, insulin_df2 = extract_data("CGM_patient2.csv", "Insulin_patient2.csv")

    parse_datetime(cgm_df2, insulin_df2)
    parse_datetime(cgm_df, insulin_df)

    meal_data, no_meal_data = extract_cgm_data(cgm_df, insulin_df)
    meal_data2, no_meal_data2 = extract_cgm_data(
        cgm_df2,
        insulin_df2,
    )
    # print(meal_data[0])
    total_meal_data = meal_data + meal_data2
    # print(len(meal_data), len(meal_data2), len(total_meal_data))
    total_no_meal_data = no_meal_data + no_meal_data2
    meal_features = extract_meal_features(total_meal_data)
    no_meal_features = extract_no_meal_features(total_no_meal_data)

    labeled_meal_features = [x + [1] for x in meal_features]
    labeled_no_meal_features = [x + [0] for x in no_meal_features]

    total_data_matrix = labeled_meal_features + labeled_no_meal_features

    train(total_data_matrix)


if __name__ == "__main__":
    main()
