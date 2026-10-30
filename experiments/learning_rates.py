import numpy as np

from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Learning Rate Experiment
# --------------------------------------------------
learning_rates = np.arange(0.001, 0.5, 0.01)

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