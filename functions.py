import scipy.stats as stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
TIME_HALF_DURATION = 15 # 10 or 15 minutes optional

def timestamp_to_quarter_hours(timestamp):
    # Convert timestamp to datetime
    event_time = datetime.fromtimestamp(timestamp)
    
    # Calculate start and end times with ±15 minutes
    start_time = event_time - timedelta(minutes=TIME_HALF_DURATION)
    end_time = event_time + timedelta(minutes=TIME_HALF_DURATION)
    
    # Calculate quarter-hour periods (0-95 for a 24-hour day)
    def time_to_quarter_hour(t):
        return t.hour * 4 + t.minute // 15
    
    start_qh = time_to_quarter_hour(start_time)
    end_qh = time_to_quarter_hour(end_time)
    
    # Create an array representing the 24-hour period divided into quarter-hours
    quarter_hours = np.zeros(96, dtype=int)
    
    if start_qh <= end_qh:
        quarter_hours[start_qh:end_qh + 1] = 1
    else:
        quarter_hours[start_qh:] = 1
        quarter_hours[:end_qh + 1] = 1
    
    return quarter_hours

def get_additional_features(timestamp):
    event_time = datetime.fromtimestamp(timestamp)
    
    # Day of the week (0=Monday, 6=Sunday)
    day_of_week = event_time.weekday()
    
    # Weekday or weekend (1=Weekend, 0=Weekday)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Season of the year (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
    month = event_time.month
    if month in [12, 1, 2]:
        season = 0  # Winter
    elif month in [3, 4, 5]:
        season = 1  # Spring
    elif month in [6, 7, 8]:
        season = 2  # Summer
    else:
        season = 3  # Autumn
    
    return [day_of_week, is_weekend, season]

def process_timestamps(df, timestamp_column):
    # Initialize an empty list to hold the combined features for each sample
    combined_features = []
    
    # Apply the function to each row
    for timestamp in tqdm(df[timestamp_column]):
        quarter_hours = timestamp_to_quarter_hours(timestamp)
        additional_features = get_additional_features(timestamp)
        combined_features.append(additional_features + quarter_hours.tolist())
    
    # Convert the list to a DataFrame or 2-D numpy array
    combined_features_df = pd.DataFrame(combined_features, columns=['day_of_week', 'is_weekend', 'season'] + [f'time_quarter_{i}' for i in range(96)])
    
    return combined_features_df


def process_location_info(df, person_id_col, lon_col, lat_col, eps=0.01, min_samples=20):
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    
    # Initialize a new column for cluster labels
    df['cluster_label'] = np.nan
    
    # Group data by `person_id`
    grouped = df.groupby(person_id_col)
    
    # Apply DBSCAN clustering for each group
    for person_id, group in tqdm(grouped):
        # Extract longitude and latitude columns
        coords = group[[lon_col, lat_col]].dropna()
        
        if len(coords) == 0:
            continue
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Create unique cluster labels
        labels = clustering.labels_
        unique_labels = {label: f"{person_id}_{label}" for label in set(labels) if label != -1}
        
        # Update the original DataFrame with cluster labels
        cluster_labels = [unique_labels[label] if label in unique_labels else -1 for label in labels]
        df.loc[coords.index, 'cluster_label'] = cluster_labels
    
    return df

def find_values_within_duration(ts, timestamps, half_duration=TIME_HALF_DURATION * 60):
    lower_bound = ts - half_duration
    upper_bound = ts + half_duration

    indices = []
    values = []

    for index, value in timestamps.items():
        if lower_bound <= value <= upper_bound:
            indices.append(index)
            values.append(value)

    return indices, values

def process_data(df, columns_of_interest, outcome_column):
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    
    # Initialize a dictionary to store label encoders and their mappings for each column
    encoders = {}
    mappings = {}
    
    for column in tqdm(columns_of_interest):
        if column==outcome_column:
            continue
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            # Initialize and fit the LabelEncoder
            encoder = LabelEncoder()
            # Fill NaN with a unique value
            df[column].fillna('NaN_placeholder', inplace=True)
            # Fit and transform the column
            df[column] = encoder.fit_transform(df[column])
            # Store the encoder for inverse transformation if needed
            encoders[column] = encoder
            # Store the mapping of original values to encoded values
            mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        else:
            # For numerical columns, fill NaN with the mean of the column
            df[column].fillna(df[column].mean(), inplace=True)
    
    # Process the outcome variable
    df[outcome_column] = df[outcome_column].apply(lambda x: 0 if pd.isna(x) else 1)
    
    return df, encoders, mappings

def parse_dates(date):
    for fmt in ('%m-%d-%Y'):
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            continue
    return pd.NaT  # Return Not a Time for unparseable formats

def wilcoxon_test_p_value(data1, data2):
    """
    Perform a Wilcoxon signed-rank test on two paired samples and return the p-value and 95% confidence interval.

    Parameters:
    - data1: First sample (list or numpy array)
    - data2: Second sample (list or numpy array)

    Returns:
    - p_value: The p-value from the Wilcoxon signed-rank test
    - confidence_interval: The 95% confidence interval of the p-value
    """
    # Perform the Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(data1, data2)
    
    # Calculate the confidence interval using the normal approximation
    z_value = stats.norm.ppf(0.995)  # 97.5th percentile point of the normal distribution
    standard_error = p_value / (2 ** 0.5)
    margin_of_error = z_value * standard_error
    
    lower_bound = max(0, p_value - margin_of_error)
    upper_bound = min(1, p_value + margin_of_error)
    
    confidence_interval = (lower_bound, upper_bound)
    
    return stat, p_value, confidence_interval

def weighted_average(values, weights):
    """
    Computes the weighted average of a list of values and weights.

    Args:
        values (list): The list of values.
        weights (list): The list of weights.

    Returns:
        float: The weighted average.
    """

    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")

    numerator = sum(value * weight for value, weight in zip(values, weights))
    denominator = sum(weights)

    return numerator / denominator
