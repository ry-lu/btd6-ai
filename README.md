# BTD6 Magic AI
![Image](https://static.wikia.nocookie.net/b__/images/8/8e/MagicBtn_1203.png/revision/latest/scale-to-width-down/243?cb=20200615102649&path-prefix=bloons)

framework for an AI that can play magic-monkey only chimps

Genetic algoirthm + neural network = magic

## Description
A neural network was trained on BTD6 round data contained in the data folder. It was only trained on magic monkeys, so it can only use magic monkeys.

 Data was collected through a DLL and the help of [BTD-Mod Helper](https://github.com/gurrenm3/BTD-Mod-Helper). The round # and magic monkey layout are inputted into the model, and the model outputs the probability that the player loses health.

The model is then used as part of the fitness function in a genetic algorithm to create the AI.

The trials in the video were all consecutive.  AI had no exposure to these two maps, only trained on data from Monkey Meadows. Videos are sped up with a mod.

## Showcase
Logs CHIMPS: [video link](https://drive.google.com/file/d/1xZ18cwrkkEFdrp4685lbqwXCJ6IVqebh/view?usp=sharing)

Cubism CHIMPS: [video link](https://drive.google.com/file/d/1EYaZ9eAFWIfQvbG2r0IfPCyWt5GWOMsr/view?usp=sharing)


## Setup
Python 3.12.2

```console
> pip install -r requirements.txt
```

## Training
yaml config format:
```yaml
epochs: 100
train_batch_size: 32
val_batch_size: 32
patience: 10
learning_rate: 0.001
val_size: 0.2
test_size: 0.5
debug: False
```

To train a new model with the configuration

```console
> python train.py -h
usage: train.py [-h] [--config CONFIG]

Train model on bloons round data.

options:
  -h, --help            show this help message and exit
  --config CONFIG, -C CONFIG
                        yaml file path for model config

> python train.py
Model: "round_predictor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                 ┃ Output Shape             ┃         Param # ┃ Connected to              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Towers (InputLayer)          │ (None, 12, 81)           │               0 │ -                         │
├──────────────────────────────┼──────────────────────────┼─────────────────┼───────────────────────────┤
│ get_item (GetItem)           │ (None, 81)               │               0 │ Towers[0][0]              │
├──────────────────────────────┼──────────────────────────┼─────────────────┼───────────────────────────┤
```
## Applying the Model
Use either model weights saved or convert to ONNX model for faster inference. Uses model and map points to simulate a game plan with genetic algorithm for 100 rounds on CHIMPS. The provided map points and possible placements are for Monkey Meadows and obtained through a custom BTD6 mod. In the showcase, a server was set up to deliver the game plan to the BTD6 mod through GET  and POST requests.

```console
> python simulate.py -h
usage: simulate.py [-h] [--map_points_path MAP_POINTS_PATH] [--possible_placements POSSIBLE_PLACEMENTS]
                   [--model_path MODEL_PATH] [--num_iteration NUM_ITERATION]
                   [--population_size POPULATION_SIZE]

Simulate a CHIMPS bloons game on a map.

options:
  -h, --help            show this help message and exit
  --map_points_path MAP_POINTS_PATH, -MAP MAP_POINTS_PATH
                        map points path from BTD6Mod
  --possible_placements POSSIBLE_PLACEMENTS, -PLACEMENT POSSIBLE_PLACEMENTS
                        sample of possible placements from BTD6Mod
  --model_path MODEL_PATH, -MODEL MODEL_PATH
                        onnx model or model weights path
  --num_iteration NUM_ITERATION, -I NUM_ITERATION
                        number of iterations for GA
  --population_size POPULATION_SIZE, -P POPULATION_SIZE
                        population size for GA

> python simulate.py
INFO: Round: 5 Win Odds: 82.55%
INFO: Towers: Druid-[0, 0, 0]- Coverage: 0.3 X: -36.05 Y: -8.19
INFO: Cost: $480

INFO: Probability of Beating Round: 0.7557235956192017
INFO: Round: 6 Win Odds: 75.57%
INFO: Towers: Druid-[0, 0, 0]- Coverage: 0.3 X: -36.05 Y: -8.19
INFO: Cost: $480
...
```


