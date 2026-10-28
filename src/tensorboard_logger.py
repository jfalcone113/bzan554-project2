from pathlib import Path
from datetime import datetime
import tensorflow as tf



# --------------------------------------------------
# Create TensorBoard Writer
# --------------------------------------------------
def create_writer(run_name, add_timestamp = True):

    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        run_name = f'{run_name}_{timestamp}'

    log_dir = Path('logs') / run_name
    log_dir.mkdir(parents = True, exist_ok = False)

    writer = tf.summary.create_file_writer((str(log_dir)))

    return writer

# --------------------------------------------------
# Log Training Metrics
# --------------------------------------------------
def log_metrics(writer, epoch, train_loss, val_loss, train_mae, val_mae):
    with writer.as_default():
        tf.summary.scalar('mse/train', train_loss, step = epoch)
        tf.summary.scalar('mse/validation', val_loss, step = epoch)
        tf.summary.scalar('mae/train', train_mae, step = epoch)
        tf.summary.scalar('mae/validation', val_mae, step = epoch)
    
    writer.flush()