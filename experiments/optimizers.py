from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Optimizer Experiment
# --------------------------------------------------

optimizers = [
    'sgd',
    'adagrad',
    'rmsprop',
    'adam'
]

# --------------------------------------------------
# Run Experiment
# --------------------------------------------------
if __name__ == '__main__':
    for optimizer in optimizers:
        experiment_config = configuration.copy()

        experiment_config['optimizer'] = optimizer

        run_experiment(
            name = f'optimizer_{optimizer}',
            configuration = experiment_config
        )