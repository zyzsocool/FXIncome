import pandas as pd
import os
import matplotlib.pyplot as plt
from fxincome import const, logger
from ydata_profiling import ProfileReport

ROOT_PATH = const.PATH.STRATEGY_POOL


def analyze_euclidean():
    src_name = "similarity_matrix_euclidean.csv"
    data = pd.read_csv(os.path.join(ROOT_PATH, src_name)).drop(columns=["date"])
    data = data.dropna().reset_index(drop=True)

    # Flatten the matrix to a vector
    flattend_data = pd.DataFrame(data.values.ravel(), columns=["distance"])
    profile = ProfileReport(
        flattend_data,
        title="Euclidean Distance Report",
        correlations=None,
        interactions=None,
    )
    profile.to_file(f"d:/{src_name}.html")


def analyze_cosine():
    src_name = "similarity_matrix_cosine.csv"
    data = pd.read_csv(os.path.join(ROOT_PATH, src_name)).drop(columns=["date"])
    data = data.dropna().reset_index(drop=True)

    # Flatten the matrix to a vector
    flattend_data = pd.DataFrame(data.values.ravel(), columns=["distance"])
    profile = ProfileReport(
        flattend_data,
        title="Cosine Distance Report",
        correlations=None,
        interactions=None,
    )
    profile.to_file(f"d:/{src_name}.html")


def analyze_features():
    src_name = "history_processed.csv"
    data = pd.read_csv(os.path.join(ROOT_PATH, src_name), parse_dates=["date"])
    data = data[const.HistorySimilarity.FEATURES + ["yield_chg_fwd_10"]]
    data["t10_fwd_direction"] = data["yield_chg_fwd_10"].apply(
        lambda x: 1 if x > 0 else 0
    )
    data = data.dropna().reset_index(drop=True)
    profile = ProfileReport(
        data,
        title="Features Report",
        correlations=None,
        interactions=None,
    )
    profile.to_file(f"d:/{src_name}.html")


def distance_histogram(distance_type: str):
    src_name = f"similarity_matrix_{distance_type}.csv"
    data_path = os.path.join(const.PATH.STRATEGY_POOL, src_name)
    distance_df = pd.read_csv(data_path)

    # Melt the DataFrame to have each date pair in a separate row
    distance_df = distance_df.melt(
        id_vars="date", var_name="date_2", value_name="distance"
    )

    # Calculate the days between each date pair
    distance_df["date"] = pd.to_datetime(distance_df["date"])
    distance_df["date_2"] = pd.to_datetime(distance_df["date_2"])
    distance_df["days_between"] = (distance_df["date_2"] - distance_df["date"]).dt.days

    # Only consider date pairs where the second date is later than the first date
    distance_df = distance_df[distance_df["days_between"] > 0]
    # Divide the distances into 100 scopes based on the days between
    distance_df["scope"] = pd.cut(distance_df["days_between"], bins=100)

    # Calculate the average distances of each scope
    avg_distances = distance_df.groupby("scope", observed=False)["distance"].mean()

    # Plot the average distances
    plt.figure(figsize=(10, 6))
    avg_distances.plot(kind="bar")
    plt.title(f"Average {distance_type} distances for each scope")
    plt.xlabel("Scope (days between)")
    plt.ylabel("Average distance")
    plt.show()


def compare_inverse_weights(distance_type: str):
    src_name = f"similarity_matrix_{distance_type}.csv"
    data_path = os.path.join(const.PATH.STRATEGY_POOL, src_name)
    distance_df = pd.read_csv(data_path)

    # Melt the DataFrame to have each date pair in a separate row
    distance_df = distance_df.melt(
        id_vars="date", var_name="date_2", value_name="distance"
    )

    # Calculate the days between each date pair
    distance_df["date"] = pd.to_datetime(distance_df["date"])
    distance_df["date_2"] = pd.to_datetime(distance_df["date_2"])
    distance_df["days_between"] = (distance_df["date_2"] - distance_df["date"]).dt.days

    # Only consider date pairs where the second date is later than the first date
    distance_df = distance_df[distance_df["days_between"] > 0]

    distance_df["weight_original"] = distance_df["distance"] / distance_df["distance"].sum()
    for c in [0, 0.1, 0.5, 1, 5, 10]:
        weights = 1 / (distance_df["distance"] + c)
        weights = weights / weights.sum()
        distance_df[f"weight_{c}"] = weights
    distance_df = distance_df[[f"weight_{c}" for c in [0, 0.1, 0.5, 1, 5, 10]] + ["weight_original"]]
    distance_df = distance_df * 1e5
    profile = ProfileReport(
        distance_df,
        title="Different Inverse Weights",
        correlations=None,
        interactions=None,
    )
    profile.to_file(f"d:/different_inverse_weights_{distance_type}.html")


def check_date(distance_type: str):
    src_name = f"similarity_matrix_{distance_type}.csv"
    data_path = os.path.join(const.PATH.STRATEGY_POOL, src_name)
    distance_df = pd.read_csv(data_path)
    # Check if each date in 'date_1' is also present in 'date_2'
    date_check_1 = distance_df["date_1"].isin(distance_df["date_2"])
    date_check_2 = distance_df["date_2"].isin(distance_df["date_1"])
    # Print the result
    print(f"All date_1 are in date_2: {date_check_1.all()}")
    print(f"All date_2 are in date_1: {date_check_2.all()}")


def analyze_predictions():
    src_name = "predictions.csv"
    data = pd.read_csv(os.path.join(ROOT_PATH, src_name)).drop(columns=["date"])
    data = data.dropna().reset_index(drop=True)

    profile = ProfileReport(
        data,
        title="Predictions Report",
        correlations=None,
        interactions=None,
    )
    profile.to_file(f"d:/{src_name}.html")


# distance_histogram("euclidean")
# check_date("cosine")
# analyze_features()
# analyze_euclidean()
# analyze_cosine()
# compare_inverse_weights("euclidean")
analyze_predictions()