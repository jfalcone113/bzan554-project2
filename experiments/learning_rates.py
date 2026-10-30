import numpy as np

from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Learning Rate Experiment
# --------------------------------------------------
learning_rates = [
    0.00001,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.1
]

# --------------------------------------------------
# Run Experiments
# --------------------------------------------------
if __name__ == '__main__':
    for learning_rate in learning_rates:
        experiment_config = configuration.copy()

        experiment_config['learning_rate'] = float(learning_rate)

        run_experiment(
            name = f'learning_rate_{learning_rate}',
            configuration = experiment_config
        )