from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Batch Size Experiment
# --------------------------------------------------
batch_sizes = [32,128,256,512,1024]

# --------------------------------------------------
# Run Experiments
# --------------------------------------------------
if __name__ == '__main__':
    for batch_size in batch_sizes:
        experiment_config = configuration.copy()

        experiment_config['batch_size'] = batch_size

        run_experiment(
            name = f'batch_size_{batch_size}',
            configuration = experiment_config
        )
