import csv
import json
import time

from datetime import datetime
from pathlib import Path

from src.train import train_model
from src.evaluate import evaluate_model


# --------------------------------------------------
# Paths
# --------------------------------------------------
results_dir = Path('results')
history_dir = results_dir / 'histories'
metrics_file = results_dir / 'metrics.csv'


# --------------------------------------------------
# Run experiment
# --------------------------------------------------
def run_experiment(name, configuration):
    results_dir.mkdir(exist_ok = True)
    history_dir.mkdir(exist_ok = True)

    # Unique experiment identifier
    run_id = (
        f'{name}_'
        f'{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
    )

    print(f'\nRunning Experiment: {run_id}')

    # Train model
    start_time = time.perf_counter()

    model, history, statistics = train_model(
        **configuration,
        run_id = run_id
    )

    training_time = time.perf_counter() - start_time

    # Evaluate validation performance
    metrics = evaluate_model(
        model,
        statistics,
        split = 'val'
    )

    # Record results
    best_epoch = min(
        range(len(history['val_loss'])),
        key = lambda i: history['val_loss'][i]
    ) + 1

    best_val_loss = min(history['val_loss'])

    result = {
        'run_id': run_id,
        'experiment': name,
        **configuration,
        'training_time': training_time,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'n': metrics['n'],
        'mse': float(metrics['mse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2'])
    }

    # Save training history
    history_path = history_dir / f'{run_id}.json'

    with open(history_path, 'w') as file:
        json.dump(history, file, indent = 4)

    # Save experiment metrics
    file_exists = metrics_file.exists()

    with open(metrics_file, 'a', newline = '') as file:

        writer = csv.DictWriter(
            file,
            fieldnames = result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
    
    print(f'\nExperiment Complete: {run_id}')
    print(f'Validation MSE: {metrics["mse"]:.4f}')
    print(f'Validation MAE: {metrics["mae"]:.4f}')
    print(f'Validation R²: {metrics["r2"]:.4f}')

    return result

