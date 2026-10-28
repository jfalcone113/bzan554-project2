import numpy as np

from .data import data_generator, target


def evaluate_model(model, statistics, split = 'val'):
    count = 0
    squared_error = 0
    absolute_error = 0.0
    mean_y = 0.0
    total_variation = 0.0

    y_min = statistics['minimums'][target]
    y_range = max(
        statistics['maximums'][target] - y_min, 1e-8
    )

    for X, y in data_generator(split, statistics):
        predictions = model.predict_on_batch(X).reshape(-1)

        # Restore original quantity scale
        actual = y.astype(np.float64) * y_range + y_min
        predicted = predictions.astype(np.float64) * y_range + y_min

        errors = actual - predicted
        squared_error += np.sum(errors ** 2)
        absolute_error += np.sum(np.abs(errors))

        # Combine chunk statistics to calculate full split r squared
        chunk_count = len(actual)
        chunk_mean = actual.mean()
        chunk_variation = np.sum((actual - chunk_mean) ** 2)

        new_count = count + chunk_count
        delta = chunk_mean - mean_y

        total_variation += (
            chunk_variation + delta ** 2 * count * chunk_count / new_count
        )
        mean_y += delta * chunk_count / new_count
        count = new_count

    if count == 0:
        raise ValueError(f'No observations in {split} split.')
    return {
        'n': count,
        'mse': squared_error / count,
        'mae': absolute_error / count,
        'r2': (1 - squared_error / total_variation if total_variation > 0 else np.nan)
    }

