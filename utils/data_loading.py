import tensorflow as tf
import numpy as np
import pandas as pd
from utils.tower_ranges import TOWER_NAMES


def preprocess(round_data, bloons_per_round):
  """ Preprocesses the data into features and labels.
  Args:
    df: dataframe of round data
  Returns:
    X: a list of numpy arrays representing features
    y: a numpy array representing labels 
  """
  X = []
  y = round_data['Health Lost'].apply(lambda x: 0 if x <0 else 1)
  towers_list = round_data['Towers'].tolist()
  map_name_list = round_data['Map Name'].tolist()
  round_list = round_data['Round'].tolist()

  tower_input_list=[]
  other_input_list=[]
  
  for towers_in_round, map_name, round_number in zip(towers_list,map_name_list,round_list):
    assert round_number < 100
    other_input_list.append(bloons_per_round[round_number])
    tower_round_input_list = []

    for tower in towers_in_round:
      if len(tower['name'].split('-'))>1:
        upgrade_multi_hot=[]
        for upgrade_path in tower['name'].split('-')[1]:
          upgrade_multi_hot+=[1]*int(upgrade_path)+[0]*(5-int(upgrade_path))
        tower_round_input_list.append([0]*(TOWER_NAMES.index(tower['name'].split('-')[0])*16) + [1] + upgrade_multi_hot +[0]*((4-TOWER_NAMES.index(tower['name'].split('-')[0]))*16)+[tower['Coverage']/100])
      else:
        tower_round_input_list.append([0]*(TOWER_NAMES.index(tower['name'].split('-')[0])*16)+[1]+[0]*15+[0]*((4-TOWER_NAMES.index(tower['name'].split('-')[0]))*16)+[tower['Coverage']/100])
         
    for i in range(12-len(tower_round_input_list)): # Populate missing towers with 0s to keep shape
      tower_round_input_list.append([0]*81)

    assert len(tower_round_input_list)<13
    tower_input_list.append(tower_round_input_list)
  try:
    X = [np.array(other_input_list).astype(np.float32), np.array(tower_input_list).astype(np.float32)]
  except:
    print(np.array(tower_input_list))

  return X, np.array(y)

class Batch_Generator(tf.keras.utils.Sequence):
  def __init__(self, df, batch_size, bloons_per_round):
    self.df = df
    self.bloons_per_round = bloons_per_round
    self.batch_size = batch_size

  def __len__(self):
    self.df = self.df.sample(frac=1)
    return int(np.ceil(len(self.df) / float(self.batch_size)))

  def __getitem__(self, index):
    # Grab a random batch of size self.batch_size from df and preprocess it
    X, y = preprocess(self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size],self.bloons_per_round)
    return (X[0],X[1]), y