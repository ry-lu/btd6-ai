# Class to represent each tower's range on a map.

import numpy as np
import pandas as pd

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def distance_squared(self, other):
      dx = self.x - other.x
      dy = self.y - other.y
      return dx*dx + dy*dy

  def interpolate(self, other, start_distance):
    """ Returns a list of points interpolated from two points
    and the magnitude of the next starting vector.
    """
    interpolated_points =[]
    vector = np.array([other.x - self.x, other.y-self.y])
    distance = np.linalg.norm(vector)
    if distance ==0:
      return [], start_distance
    vector_norm = vector/distance
    start_vector = np.zeros(2)+vector_norm*start_distance
    # Take steps with a vector of magnitude 4      
    while np.linalg.norm([start_vector[0], start_vector[1]]) < np.linalg.norm([other.x-self.x, other.y-self.y]):
      vector_to_point = Point(start_vector[0]+self.x, start_vector[1]+self.y)
      interpolated_points.append(vector_to_point)
      start_vector+=vector_norm*4


    return interpolated_points, np.linalg.norm([start_vector[0], start_vector[1]]) - np.linalg.norm([other.x-self.x, other.y-self.y])

class TowerRange:
  """ Class that represents a tower's ranges.
  Args:
    base_range: int, the base range of the tower
    top_range: list of length 5, range given by top path upgrades
    middle_range: list of length 5, range given by middle path upgrades
    bottom_range: list of length 5, range given by bottom path upgrades
  """
  def __init__(self, base_range, top_range,middle_range,bottom_range):
    assert len(top_range) == 6 and len(middle_range) == 6 and len(bottom_range) == 6
    self.base_range = base_range
    self.top_range = top_range
    self.middle_range = middle_range
    self.bottom_range = bottom_range

  def find_tower_coverage(self,upgrade_path, tower_position, map_points):
    """ Finds the # times a tower attacks given a 6 baloon simulation along a map.
    Args:
      upgrade_path: 3 char string (EX: "001")
      tower_position: point, tower position
      map_points: list of points of a map path
    Returns:
      num of points
    """
    top_path, middle_path, bottom_path = int(upgrade_path[0]), int(upgrade_path[1]),int(upgrade_path[2])
    assert top_path < 6 and middle_path < 6 and bottom_path < 6 and len(upgrade_path) == 3
    tower_range = self.base_range+self.top_range[top_path]+self.middle_range[middle_path]+self.bottom_range[bottom_path]
  
    points_in_range = []
    for idx, point in enumerate(map_points):
      if point.distance_squared(tower_position) < (tower_range+5)*(tower_range+5):
        points_in_range.append(idx)

    num_attacks = 0
    attack_cooldown = 21
    for idx, point in enumerate(map_points):
      attack_cooldown+=1
      if attack_cooldown > 20:
        for i in range(6):
          if idx-i*70 > 0 and idx-i*70 in points_in_range:
            num_attacks+=1
            attack_cooldown = 0
            break

    map_length =  len(map_points)
    for idx in range(200):
      attack_cooldown+=1
      if attack_cooldown > 20:
        for i in range(6):
          if -200+idx+i*70 < 0 and map_length -200+idx + i*70 in points_in_range:
            num_attacks+=1
            attack_cooldown = 0
            break

    return num_attacks
  

TOWER_NAMES = ["WizardMonkey","Alchemist","NinjaMonkey","Druid","SuperMonkey"]
UPGRADE_PATHS = ["000","001","010","100","002","020","200","101","110","011","201","102","210","120","012","021","220","202","022","032","023","302","203","320","230","042","024","204","402","240","420","052","025","502","205","520","250","031","013","301","103","310","130","041","014","104","401","140","410","051","015","501","105","510","150","030","003","300","040","004","400","050","005","500"]
TOWER_RANGES = {'WizardMonkey': TowerRange(40,[0,0,0,20,20,30],[0,0,0,10,10,20],[0,0,0,20,20,20]), 
                'Alchemist':TowerRange(45,[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]), 
                'NinjaMonkey':TowerRange(40,[0,7,7,7,7,18],[0,0,0,0,0,5],[0,0,0,0,0,0]),
                'Druid':TowerRange(35,[0,0,0,0,0,0],[0,0,0,0,10,10],[0,10,10,10,10,15]),
                'SuperMonkey':TowerRange(50, [0,0,0,0,15,15], [0,10,22,22,22,32], [0,0,0,3,3,3])}
