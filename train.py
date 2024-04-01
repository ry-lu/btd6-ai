import json
import math
import os
import logging
import argparse
import yaml
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np

from utils.data_preparation import *
from utils.data_loading import Batch_Generator
from model.model import create_model

bloon_rounds_path = Path('./data/btd6-bloon-rounds.csv')
map_file_path = Path('./data/map_points.json')
round_data_path = Path('./data/magic-meadows-round-outcomes.gz')
standard_data_path = Path('./configs/standard_config.yaml')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        config_name,
        epochs: int = 5,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        learning_rate: float = 1e-5,
        patience: int = 15,
        val_size: float = 0.2,
        test_size: float = 0.5,
        debug: bool = False,
):
    model.summary()
    # Load and prepare data
    bloons_per_round=get_bloons_per_round(bloon_rounds_path)
    map_points = get_map_points(map_file_path)
    logging.info("Preparing Data...")
    round_data = prepare_data(round_data_path, map_points, debug)
    round_data_train, round_data_test, round_data_val = split_data(round_data,val_size,test_size)

    # Create batch generators
    train_loader = Batch_Generator(round_data_train, train_batch_size, bloons_per_round)
    val_loader = Batch_Generator(round_data_val, val_batch_size, bloons_per_round)
    test_loader = Batch_Generator(round_data_test, val_batch_size, bloons_per_round)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Train Batch size:{train_batch_size}
        Val Batch size:  {val_batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(round_data_train)}
        Validation size: {len(round_data_val)}
        Patience:        {patience}
    ''')

    # Keras Training
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, restore_best_weights=True)
    history = model.fit(train_loader, epochs=epochs, validation_data=(val_loader), callbacks=[early_callback])
    
    logging.info('Evaluating on test set...')
    model.evaluate(test_loader)

    # Save model checkpoint
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    tf.keras.models.save_model(model, dir_checkpoint / f'{config_name}_checkpoint.keras')
    logging.info(f'Checkpoint saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train model on bloons round data.')
    parser.add_argument('--config', '-C', type=str, default=standard_data_path, help='yaml file path for model config')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    np.random.seed(1)
    model = create_model()
    train_model(
        model=model,
        config_name=os.path.splitext(os.path.basename(args.config))[0],
        epochs=config['epochs'],
        train_batch_size=config['train_batch_size'],
        val_batch_size=config['val_batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        val_size=config['val_size'],
        test_size=config['test_size'],
        debug=config['debug'],
    )