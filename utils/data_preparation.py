import json
from tqdm import tqdm
import pandas as pd
from utils.tower_ranges import TOWER_NAMES, TOWER_RANGES, Point
from utils.tower_encodings import TOWERS
from sklearn.model_selection import train_test_split

INCOME_PER_ROUND=[650,650,650,650,650,650,813,995,1195,1394,1708,1897,2089,2371,2630,2896,3164,3329,3687,3947,4133,4484,
        4782,5059,5226,5561,5894,6556,6822,7211,7548,8085,8712,8917,9829,10979,11875,13214,14491,16250,
        16771,18952,19611,20889,22183,24605,25321,26958,29801,34559,37575,38673.5,40269,41193.5,43391,45874,
        47160,49019,51317,53476,54399,55631,57017,59843,60693,63764,64769,65792,66570,67961,70580,72083,
        73587,74979,78023,80691,82007,84547,89409,96118,97518,102884,107641,112390,119434,122060,123008,
        125635, 128949, 131120,131460,135651,140188,142135,149802,153520,163475,164893,174546,177374,178909]

def get_map_points(map_file_path):
  """Converts map points file in BTD6 mod format to 
  point format and interpolates the gaps."""
  with open(map_file_path, 'r') as file:
    map = json.loads(file.read())
  points=[]
  starting_distance = 0
  for idx, point in enumerate(map):
    if idx != len(map)-1 and idx !=0:
      point1 = Point(point['x'], point['y'])
      point2 = Point(map[idx+1]['x'], map[idx+1]['y'])
      # Find points in between
      temp, starting_distance = point1.interpolate(point2, starting_distance)
      points.extend(temp)
  return points

def get_bloons_per_round(bloon_rounds_path):
  """Loads bloons data."""
  bloon_rounds = pd.read_csv(bloon_rounds_path)
  bloon_rounds_dict = bloon_rounds.to_dict()
  return [[bloon_rounds_dict[bloon][round]/50 for bloon in bloon_rounds.columns if bloon!='Round' and bloon!='RBE'] for round in range(100)]


def get_possible_placements(possible_placements_path, map_points):
  """Loads a list of possible placements retreived and sampled
  from mod."""
  with open(possible_placements_path, 'r') as file:
    possible_placements = json.loads(file.read())
  return find_my_placement_points(possible_placements, map_points)

def find_my_placement_points(possible_placements, map_points):
  """Returns a list of 12 of the best placement points."""
  possible_coverage = []
  # Calculate how much coverage each placement point provides
  for possible_placement in possible_placements:
    possible_placement_point = Point(possible_placement['x'],possible_placement['y'])  
    possible_coverage.append([
        possible_placement_point,
        TOWERS['Druid']['Range'].find_tower_coverage(
            '000', 
            possible_placement_point, 
            map_points
        )
    ])
  # Sort points by best coverage
  possible_coverage.sort(key=lambda x:x[1], reverse = True)
  my_placement_points = []
  for coverage in possible_coverage:
    covered = False
    if len(my_placement_points) == 13:
      break
    for placement_point in my_placement_points:
      # Ensure towers when placed don't overlap
      if placement_point.distance_squared(coverage[0]) < 121 or placement_point.distance_squared(coverage[0]) > 10000:
        covered = True
    if not covered:
      #print(coverage[1])
      my_placement_points.append(coverage[0])
  return my_placement_points[1:], my_placement_points[0]

def clean_data(df):
  """Remove outlier data and those that
  do not fit in our scheme."""
  drop_indexes = {}
  for i, (towers_per_round, round_num, towers) in enumerate(zip(df['Towers'].tolist(),df['Round'].tolist(),df['Towers'].tolist())):
    if round_num > 100 or len(towers_per_round)>12:
      drop_indexes[i] = 1
    for tower in towers:
      if tower['name'] == "PermaPhoenix":
        tower['name'] = "WizardMonkey-050"
      elif tower['name'].split('-')[0] not in TOWER_NAMES:
        drop_indexes[i] = 1
        break
  return df.drop(drop_indexes).reset_index(drop=True)

def add_tower_coverage(towers, map_points):
  """ Adds key-value pair Coverage to each tower in list of towers
  """
  for tower in towers:
    if len(tower['name'].split('-')) == 2:
      tower_base_name = tower['name'].split('-')[0]
      tower_upgrade_path = tower['name'].split('-')[1]
    else:
      tower_base_name = tower['name']
      tower_upgrade_path = '000'
    tower_position = Point(tower['position']['x'], tower['position']['y'])
    tower["Coverage"] = TOWER_RANGES[tower_base_name].find_tower_coverage(tower_upgrade_path, tower_position, map_points)
  return towers

def prepare_data(round_data_path, map_points, debug):
    round_data = pd.read_csv(round_data_path, compression='gzip').drop_duplicates().reset_index(drop=True)
    # Use less data if in debug mode
    if debug:
      round_data = round_data[:100]

    round_data['Towers'] = round_data['Towers'].apply(json.loads)
    round_data = clean_data(round_data)

    tqdm.pandas(desc="Adding tower coverage")
    round_data['Towers'] = round_data['Towers'].progress_apply(lambda x: add_tower_coverage(x, map_points))
    return round_data

def split_data(df, val_size,test_size):
  """ Splits data into train, val, and test.
  Args:
    df: dataframe of round data
    val_size: percentage of data to allocate as val data
    test_size: percentage of val data to allocate as test data
  Returns:
    train, val, test: dataframes of round data
  """
  round_data_train, round_data_test = train_test_split(df, test_size = val_size)
  round_data_test, round_data_val = train_test_split(round_data_test, test_size = test_size)
  return round_data_train.reset_index(drop=True),round_data_test.reset_index(drop=True),round_data_val.reset_index(drop=True)