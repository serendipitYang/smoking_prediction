import pandas as pd
import numpy as np
import random
import string

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define constants
NUM_ROWS = 50000
UNIQUE_PERSONS = 50
PERSON_ID_PREFIX = 'id'
SMOKING_SUBSTANCES = ['cigarette', 'E-Cigarette', 'Nicotine pouch (e.g. On!, Zyn)', 'Cigar / Little Cigar / Cigarillo', 'Pipe filled with tobacco', 'Other']
NON_SMOKING_RATE = 0.8  # Rate of non-smoking entries (NaN)
LONGITUDE_MIN, LONGITUDE_MAX = -122.64201806, -68.73344728
LATITUDE_MIN, LATITUDE_MAX = 26.362845, 45.8375551
TIMESTAMP_MIN, TIMESTAMP_MAX = 1614203840, 1706642573
MAY_11_2023_TIMESTAMP = 1683763200  # May 11, 2023
MAX_DURATION_DAYS = 20

# Generate person_id list
person_ids = [PERSON_ID_PREFIX + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)) for _ in range(UNIQUE_PERSONS)]

# Generate data for each person
data = []

for person_id in person_ids:
    # Random sample size for each person between 100 and 2000
    sample_size = random.randint(100, 2000)
    
    # Generate timestamps within a duration of MAX_DURATION_DAYS
    start_time = random.randint(TIMESTAMP_MIN, TIMESTAMP_MAX - MAX_DURATION_DAYS * 24 * 60 * 60)
    end_time = start_time + random.randint(0, MAX_DURATION_DAYS * 24 * 60 * 60)
    timestamps = np.random.uniform(start_time, end_time, sample_size).astype(int)

    # Generate longitude and latitude, ensuring cluster-ability
    center_longitude = np.random.uniform(LONGITUDE_MIN, LONGITUDE_MAX)
    center_latitude = np.random.uniform(LATITUDE_MIN, LATITUDE_MAX)
    longitudes = np.random.normal(center_longitude, 0.5, sample_size)
    latitudes = np.random.normal(center_latitude, 0.5, sample_size)

    # Generate smoking substance data
    substances = []
    for _ in range(sample_size):
        if random.random() < NON_SMOKING_RATE:
            substances.append(np.nan)
        else:
            if len(substances) < 7025:
                substances.append('cigarette')
            else:
                substances.append(random.choice(SMOKING_SUBSTANCES[1:]))
    
    # Generate Is_after_covid column
    is_after_covid = [1 if ts > MAY_11_2023_TIMESTAMP else 0 for ts in timestamps]

    # Append data
    person_data = list(zip([person_id] * sample_size, timestamps, longitudes, latitudes, substances, is_after_covid))
    data.extend(person_data)

    # Check if data exceeds the required rows
    if len(data) >= NUM_ROWS:
        break

# Truncate data to NUM_ROWS
data = data[:NUM_ROWS]

# Create DataFrame
columns = ['person_id', 'timestamp', 'longitude', 'latitude', 'substance', 'is_after_covid']
df = pd.DataFrame(data, columns=columns)

# df.head()
df.to_csv('./synthetic_raw_data.csv',index=False)
