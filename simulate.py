# Simulates a tower defense plan with given a set of map points and model.

import logging
import argparse
from pathlib import Path

from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np
import onnxruntime as ort
from model.model import create_model

from utils.data_preparation import *
from utils.tower_encodings import (
    TOWER_ORDER,
    create_tower_encoding,
    create_tower_encodings_restricted,
    find_tower_similiarity
)
from genetic import tower_geneticalogirthm

bloon_rounds_path = Path('./data/btd6-bloon-rounds.csv')
map_file_path = Path('./data/map_points.json')
round_data_path = Path('./data/magic-meadows-round-outcomes.gz')
possible_placements_path = Path('./data/possible_placements.json')
standard_data_path = Path('./configs/standard_config.yaml')
model_path = Path('./checkpoints/standard_config_checkpoint.onnx')

def predict(fnn_model, otherInputs, towerInputs):
    # Check if onnx or keras model
    if isinstance(fnn_model, ort.InferenceSession):
        result = fnn_model.run(None, {
        'Other': np.reshape(otherInputs,(1,46)), 
        'Towers':np.reshape(towerInputs,(1,12,81))})[0]
    else:
        result = fnn_model(
        (np.reshape(otherInputs,(1,46)),
            np.reshape(towerInputs,(1,12,81))))
    
    return result

def fitness_function(tower_comp, round_num, tower_encodings, goal, model, bloons_per_round):
  """Calculates fitness of a 12 set tower combination.
  Fitness is based on the model output, cost, and other heurstics. 
  A large penalty is applied to combinations that are impossible."""
  win_odds = 0
  cost = 0
  tower_count = 0
  tower_list = []
  tower_tier_five_list=[]

  # Calculate tower combination cost
  for tower_idx, tower in enumerate(tower_comp):
    if tower < 0:
      tower_list+=[0]*81
    else:
      tower_encoding = tower_encodings[tower_idx][int(tower)]
      if tower_encoding.tier_5:
        tower_tier_five_list.append(tower_encoding.tier_5)
      tower_count+=1
      cost+= tower_encoding.cost
      tower_list+=tower_encoding.encoding[tower_idx]

  if cost > INCOME_PER_ROUND[round_num]:
    # tower combination is too expensive, reduce cost
    return cost-INCOME_PER_ROUND[round_num]+1000+tower_count/12*8
  
  # tier-5 towers good
  if len(tower_tier_five_list) != len(set(tower_tier_five_list)):
    return 1000+0.4+3.5
  
  # Find probability towers beat round
  win_odds = predict(
    fnn_model, 
    np.array(bloons_per_round[round_num]).astype(np.float32),
    np.array(tower_list).astype(np.float32))
  
  if win_odds > .7:
      # Encourage save money for high-tier towers
      loss = -800 + tower_count/12*400 - len(tower_tier_five_list)*3000
      # Encourage diversity in tower combination
      loss-= find_tower_similiarity(tower_comp, tower_encodings, goal)
      loss-=(INCOME_PER_ROUND[round_num]-cost)/30
  else:
    # If predicted to lose, fitness low
    loss = -win_odds*3

  return loss


def generate_tower_combination(
    fnn_model,
    current_round, 
    previous_towers, 
    algorithm_param, 
    goal, 
    bloons_per_round,
    possible_placements,
    tower_encoding,
):
  """Uses GA to find optimal tower combination for a sequence
  of rounds."""
  tower_encodings_restricted = create_tower_encodings_restricted(
    INCOME_PER_ROUND[current_round], 
    previous_towers,tower_encoding)

  varbound = [-1]*12

  # Bound our tower combinations by the restrictions
  for idx, previous_tower in enumerate(previous_towers):
    if hasattr(previous_tower, 'name'):
      varbound[idx] = [0,len(tower_encodings_restricted[idx])-1]
    else:
      varbound[idx] = [-300,len(tower_encodings_restricted[idx])-1]

  model=tower_geneticalogirthm(function=fitness_function,dimension=12, 
           variable_type='int',variable_boundaries=np.array(varbound), 
           algorithm_parameters = algorithm_param, 
           convergence_curve = False, 
           progress_bar = False,
           round_num = current_round,
           tower_encodings = tower_encodings_restricted,
           goal = goal,
           bloons_per_round=bloons_per_round,
           fnn_model=fnn_model)
  
  model.run()

  # Record tower combination into tower_json and tower_name
  cost = 0
  tower_encoding_list = []
  towers_list = [0]*12
  tower_names = []
  tower_json = []

  for tower_idx, tower in enumerate(model.best_variable):
    if tower >= 0:
      tower_encoding = tower_encodings_restricted[tower_idx][int(tower)]
      towers_list[tower_idx] = tower_encoding
      tower_encoding_list.append(tower_encoding.encoding[tower_idx])
      tower_json.append({
         'Tower':TOWER_ORDER.index(tower_encoding.name), 
         'Upgrade':tower_encoding.upgrade_path ,
         'Position':{'x':possible_placements[tower_idx].x, 
                     'y':possible_placements[tower_idx].y}, 
                     'id':tower_idx})
      tower_names.append(
        f"{tower_encoding.name}-{tower_encoding.upgrade_path}- "
        f"Coverage: {tower_encoding.encoding[tower_idx][-1]} "
        f"X: {possible_placements[tower_idx].x:.2f} "
        f"Y: {possible_placements[tower_idx].y:.2f}"
      )       
      cost+=tower_encoding.cost
    else:
      tower_encoding_list.append([0]*81)
      towers_list[tower_idx] = -1

  # One last check to see if tower combination will beat the round
  win_odds = predict(
    fnn_model,
    np.array(bloons_per_round[current_round]).astype(np.float32),
    np.array(tower_encoding_list).astype(np.float32))
  
  current_round_temp=current_round
  logging.info(f'Round: {current_round_temp} Win Odds: {win_odds[0][0]*100:.2f}%')
  logging.info("Towers: " +','.join(tower_names))
  logging.info(f'Cost: ${cost}\n')

  if win_odds < 0.7:
    return towers_list, current_round_temp, tower_json
      
  while win_odds >= 0.7 and current_round_temp < 99:
    current_round_temp+=1
    # See how many rounds the tower comboination can last
    win_odds = predict(
       fnn_model, 
       np.array(bloons_per_round[current_round_temp]).astype(np.float32),
       np.array(tower_encoding_list).astype(np.float32))
    if win_odds>0.7:
      logging.info(f'Round: {current_round_temp} Win Odds: {win_odds[0][0]*100:.2f}%')
      logging.info("Towers: " +','.join(tower_names))
      logging.info(f'Cost: ${cost}\n')

  return towers_list, current_round_temp-1, tower_json

def get_args():
    parser = argparse.ArgumentParser(description='Simulate a CHIMPS bloons game on a map.')
    parser.add_argument(
       '--map_points_path', 
       '-MAP', type=str, default=map_file_path, 
       help='map points path from BTD6Mod')
    parser.add_argument(
       '--possible_placements', 
       '-PLACEMENT', type=str, default=possible_placements_path, 
       help='sample of possible placements from BTD6Mod')
    parser.add_argument(
       '--model_path', 
       '-MODEL', type=str, 
       default=model_path, 
       help='onnx model or model weights path')

    
    parser.add_argument('--num_iteration', '-I', type=int, default=25, help='number of iterations for GA')
    parser.add_argument('--population_size', '-P', type=int, default=50, help='population size for GA')

    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Get data
    map_points = get_map_points(args.map_points_path)
    bloons_per_round = get_bloons_per_round(bloon_rounds_path)
    possible_placements, hero_placement = get_possible_placements(args.possible_placements,map_points)
    all_tower_encoding = create_tower_encoding(possible_placements,map_points)

    np.random.seed(1)

    try:
        fnn_model = ort.InferenceSession(args.model_path)
    except Exception:
        try:
            fnn_model = create_model(compile=False)
            fnn_model.load_weights(args.model_path)
        except Exception:
            print("Model unable to be loaded")

    # Stores the tower combinations in json
    building_plan = []

    # CHIMPS starts on round 6
    i = 5
    # We start wtih 0 towers, -1 means no tower
    previous_towers = [-1]*12

    max_num_iteration = args.num_iteration
    population_size = args.population_size

    # Generate tower combos till round 100
    while i < 100:
        algorithm_param = {'max_num_iteration': max_num_iteration,\
                    'population_size':population_size,\
                    'mutation_probability':0.2,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}


        tower_combo, round_it_lasts, tower_json = generate_tower_combination(
            fnn_model=fnn_model,
            current_round=i, 
            previous_towers=previous_towers, 
            algorithm_param=algorithm_param, 
            goal = [], 
            bloons_per_round=bloons_per_round, 
            possible_placements=possible_placements,
            tower_encoding=all_tower_encoding
        )
        # Update plan
        tower_json.append({'Tower':-1, 'Upgrade':[-1,-1,-1] ,'Position':{'x':hero_placement.x, 'y':hero_placement.y}, 'id':-1})
        building_plan.append({'Round':i, 'Towers':tower_json})
        
        if len(tower_combo) == 0:
            # If no plan, search harder
            max_num_iteration += 50
            population_size+=50
        else:
            max_num_iteration = 25
            population_size= 50
            previous_towers = tower_combo
            i=round_it_lasts+1
