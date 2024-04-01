# This module creates tower encodings for the genetic algorithm.
# A tower has a range, cost, possible upgrades, and its encoding.
# Tower encodings are used to input into the neural network model.
# We arbitrary have a maximum of 12 towers.
# At each of the 12 tower slots, we create a tower_list of
# all possible tower combinations given the constraints.

import numpy as np

from utils.tower_ranges import TowerRange, TOWER_NAMES, UPGRADE_PATHS

class TowerCost:
  """ Class that represents a tower's cost.
  Args:
    base_cost: int, the base cost of the tower
    top_cost: list of length 5, cost given by top path upgrades
    middle_cost: list of length 5, cost given by middle path upgrades
    bottom_cost: list of length 5, cost given by bottom path upgrades
  """
  def __init__(self, base_cost, top_cost,middle_cost,bottom_cost):
    assert len(top_cost) == 5 and len(middle_cost) == 5 and len(bottom_cost) == 5
    self.base_cost = base_cost
    self.top_cost = top_cost
    self.middle_cost = middle_cost
    self.bottom_cost = bottom_cost
  def find_tower_cost(self, upgrade_path):
    """ Finds tower cost based on its upgrades.
    Args:
      upgrade_path: 3 char string (EX: "001")
    Returns:
      An integer representing tower cost.
    """
    top_path, middle_path, bottom_path = int(upgrade_path[0]), int(upgrade_path[1]),int(upgrade_path[2])
    assert top_path < 6 and middle_path < 6 and bottom_path < 6 and len(upgrade_path) == 3
    return self.base_cost+sum(self.top_cost[:top_path])+sum(self.middle_cost[:middle_path])+sum(self.bottom_cost[:bottom_path])

TOWER_ORDER = ["WizardMonkey","SuperMonkey","NinjaMonkey","Alchemist","Druid"]

TOWERS = {'WizardMonkey': {'Range':TowerRange(40,[0,0,0,20,20,30],[0,0,0,10,10,20],[0,0,0,20,20,20]), 
                                 'Cost':TowerCost(450,[180,540,1560,12000,38400],[360,1140,3600,5400,64800],[360,360,1800,3360,31800])}, 
                'Alchemist':{'Range':TowerRange(45,[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]), 
                             'Cost':TowerCost(660,[300,420,1500,3600,72000],[300,570,3600,5400,54000],[780,540,1200,3300,48000])}, 
                'NinjaMonkey':{'Range':TowerRange(40,[0,7,7,7,7,18],[0,0,0,0,0,5],[0,0,0,0,0,0]),
                               'Cost':TowerCost(600,[360,420,1020,3300,42000],[420,600,1080,6240,26400],[300,480,2700,6000,48000])}, 
                'Druid':{'Range':TowerRange(35,[0,0,0,0,0,0],[0,0,0,0,10,10],[0,10,10,10,10,15]),
                         'Cost':TowerCost(480,[300,1200,1980,5400,78000],[300,420,1140,6000,42000],[120,360,720,3000,54000])}, 
                'SuperMonkey':{'Range':TowerRange(50, [0,0,0,0,15,15], [0,10,22,22,22,32], [0,0,0,3,3,3]),
                               'Cost':TowerCost(3000,[3000,3600,24000,120000,600000],[1200,1680,9600,22800,108000],[3600,1440,6720,72000,240000])}
                }

class Tower:
  """ Class that represents a specific tower.
  Args:
    name: string
    cost: integer
    upgrade_path: list of 3 ints
    encoding: list of 12 list of 81 bools, represents a tower's encoding in the 12 possible placements
  """
  def __init__(self, name, cost, upgrade_path, encoding,placement_points, map_points):
    assert len(upgrade_path) == 3 and len(encoding) == 80
    self.name=name
    self.cost = cost
    self.upgrade_path = upgrade_path
    self.encoding = []

    for placement_point in placement_points:
      upgrade_path = ''.join([str(upgrade) for upgrade in self.upgrade_path])
      self.encoding.append(encoding + [TOWERS[self.name]['Range'].find_tower_coverage(upgrade_path, placement_point, map_points)/100])

    if str(5) in upgrade_path:
      self.tier_5 = name+str(upgrade_path.index('5'))
    else:
      self.tier_5 = False

  def can_upgrade_to(self, other):
    """ Sees if other tower can upgrade to self."""
    if self.upgrade_path[0] >= other.upgrade_path[0] and  self.upgrade_path[1] >= other.upgrade_path[1] and  self.upgrade_path[2] >= other.upgrade_path[2]:
      return True
    else:
      return False
    

def create_tower_encoding(placement_points, map_points):
  """ Returns a tower list that contains all
  combinations of the towers.
  """
  tower_list=[]
  for tower_idx, tower_name in enumerate(TOWER_NAMES):
    # Remove alchemist
    if tower_name != "Alchemist":
      for upgrade in UPGRADE_PATHS:
        upgrade_multi_hot=[] 
        for upgrade_path in upgrade:                                                                 
          upgrade_multi_hot+=[1]*int(upgrade_path)+[0]*(5-int(upgrade_path))
        # Encode the towers
        tower_cost = TOWERS[tower_name]['Cost'].find_tower_cost(upgrade)
        tower_upgrade_path = [int(upgrade[0]),int(upgrade[1]),int(upgrade[2])]
        tower_encoding = [0]*(tower_idx*16)+ [1] + upgrade_multi_hot + [0]*((4-tower_idx)*16)
        tower_list.append(Tower(
          tower_name, 
          tower_cost, 
          tower_upgrade_path,
          tower_encoding,
          placement_points, 
          map_points))
  return tower_list


def create_tower_encodings_restricted(money, previous_towers, tower_encoding):
  """ Returns a list of 12 tower_lists, each list
  containing the combinations of towers
  given the restrictions (money, previous towers).
  """
  tower_encodings_list = [-1]*12
  previous_tower_cost = 0
  for idx, previous_tower in enumerate(previous_towers):
    if hasattr(previous_tower, 'name'):
      previous_tower_cost+=previous_tower.cost
      tower_list = []
      for tower in tower_encoding:
        if previous_tower.name == tower.name and tower.cost <= money and tower.can_upgrade_to(previous_tower):
          tower_list.append(tower)
      tower_encodings_list[idx] = tower_list
  
  for idx, previous_tower in enumerate(previous_towers):
    if not hasattr(previous_tower, 'name'):
      tower_list = []
      for tower in tower_encoding:
        if tower.cost <= money-previous_tower_cost:
          tower_list.append(tower)
      tower_encodings_list[idx] = tower_list
    
  return tower_encodings_list

def find_tower_similiarity(tower_comp, tower_encodings, goal):
  """A heuristic function to measure how similar two towers are."""
  bonus = 0
  found_towers = []
  for tower_idx, tower in enumerate(tower_comp):
    if tower >= 0:
      tower_encoding = tower_encodings[tower_idx][int(tower)]
      for goal_idx, goal_tower in enumerate(goal):
        if hasattr(goal_tower, 'name'):
          if goal_tower.name == tower_encoding.name and goal_tower.encoding[goal_idx] not in found_towers:
            temp = np.add(np.array(goal_tower.encoding[goal_idx]),np.array(tower_encoding.encoding[tower_idx]))
            bonus += 10*((np.count_nonzero(temp == 2)+np.count_nonzero(temp==0))-80)
            found_towers.append(goal_tower.encoding[goal_idx])
          else:
            bonus-=20
  return bonus