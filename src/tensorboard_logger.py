from datetime import datetime
from pathlib import Path
import tensorflow as tf


def create_writer(run_name = 'baseline'):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    project_dir = Path(__file__).resolve().parent.parent
    log_dir = project_dir / 'logs' / f'{run_name}_{timestamp}'

    return tf.summary.create_file_writer(str(log_dir))


def log_metrics(writer, epoch, train_loss, val_loss, train_mae, val_mae):
    with writer.as_default():
        tf.summary.scalar('mse/train', train_loss, step = epoch)
        tf.summary.scalar('mse/validation', val_loss, step = epoch)
        tf.summary.scalar('mae/train', train_mae, step = epoch)
        tf.summary.scalar('mae/validation', val_mae, step = epoch)
    
    writer.flush()