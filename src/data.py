from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Data path
data_path = Path(__file__).resolve().parent.parent/ 'data' / 'pricing.csv' # store filesystem path object of data file

# Incremental learning
chunk_size = 16384 # number of observations to load from data

# Reproducibility
random_state = 42 # random seed

# Dataset train/validation/test split
train_size = 0.7
val_size = 0.15
test_size = 0.15

# Predictor variables
numerical_features = ['price','order','duration']
categorical_features = ['category']

# Response variable
target = 'quantity'

# --------------------------------------------------
# Incremental data loading
# --------------------------------------------------

def load_chunks(filepath = data_path, chunk_size = chunk_size):
    reader = pd.read_csv(
        filepath,
        chunksize = chunk_size
    )
    
    for chunk in reader:
        yield chunk

# --------------------------------------------------
# Train/Validation/Test Split
# --------------------------------------------------

def load_split(split, filepath = data_path):
    rng = np.random.default_rng(random_state)

    for chunk in load_chunks(filepath):
        random_values = rng.random(len(chunk))

        if split == 'train':
            mask = random_values < train_size

        elif split == 'val':
            mask = (
                (random_values >= train_size)
                & (random_values < train_size + val_size)
            )
        
        elif split == 'test':
            mask = random_values >= train_size + val_size

        else:
            raise ValueError('split must be train, val, or test')
        
        yield chunk.loc[mask].copy()

# --------------------------------------------------
# Calculate Training Statistics
# --------------------------------------------------

def calculate_statistics():

    minimums = None
    maximums = None

    categories = set()

    columns = numerical_features + [target]

    for chunk in load_split('train'):

        if chunk.empty:
            continue
        
        chunk_min = chunk[columns].min()
        chunk_max = chunk[columns].max()
        
        if minimums is None:
            minimums = chunk_min
            maximums = chunk_max
        
        else:
            minimums = np.minimum(minimums, chunk_min)
            maximums = np.maximum(maximums, chunk_max)
        
        categories.update(chunk['category'].unique())
        
    if minimums is None:
        raise ValueError('No training observations found.')

    return {
        'minimums': minimums,
        'maximums': maximums,
        'categories': sorted(categories)
    }

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------


def preprocess_chunk(chunk, statistics):

    # Numerical inputs
    X_numeric = chunk[numerical_features].to_numpy(dtype = np.float32)

    min_values = statistics['minimums'][numerical_features].to_numpy()
    max_values = statistics['maximums'][numerical_features].to_numpy()

    denominator = np.maximum(max_values - min_values, 1e-8)

    X_numeric = (X_numeric - min_values) / denominator
    
    # One hot encode category using training categories
    categories = statistics['categories']
    
    X_category = (
        pd.get_dummies(chunk['category'])
        .reindex(columns = categories, fill_value = 0)
        .to_numpy(dtype = np.float32)
    )

    # Combine numerical and categorical inputs
    X = np.column_stack((X_category, X_numeric)).astype(np.float32)

    # Normalize response variable
    y_min = statistics['minimums'][target]
    y_max = statistics['maximums'][target]

    y = chunk[target].to_numpy(dtype = np.float32)

    y = (y - y_min) / max(y_max - y_min, 1e-8)

    return X, y.astype(np.float32)

# --------------------------------------------------
# Prepared Data Generator
# --------------------------------------------------

def data_generator(split, statistics):
    
    for chunk in load_split(split):

        if chunk.empty:
            continue

        X, y = preprocess_chunk(chunk, statistics)

        yield X, y

# --------------------------------------------------
# Test Data Pipeline
# --------------------------------------------------

if __name__ == '__main__':
    print('Calculating training stats...')

    statistics = calculate_statistics()

    print('\nTraining Minimums:')
    print(statistics['minimums'])

    print('\nTraining Maximums:')
    print(statistics['maximums'])

    print('\nNumber of categories:')
    print(len(statistics['categories']))

    print('\nLoading first training chunk...')

    X, y = next(data_generator('train', statistics))

    print('X shape:', X.shape)
    print('y shape:', y.shape)

    print('\n First Observation:')
    print(X[0])

    print('\nFirst Response:')
    print(y[0])
