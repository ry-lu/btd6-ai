from keras.models import Model
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Concatenate, Flatten, Embedding
from keras import activations
import tensorflow as tf

class DensePermutation(keras.layers.Layer):
  """ Layer that deals with permutation invariance using weight sharing.
  """
  def __init__(self, units, activation=None, name=None, **kwargs):
    super(DensePermutation, self).__init__(name=name, **kwargs)
    self.units = units
    self.activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    self.my_weight = self.add_weight(
          shape=(input_shape[0][-1], self.units),
          initializer="random_normal",
          name="self_kernel",
          trainable=True,
      )
   
  def call(self, inputs):
    outputs = 0
    for input in inputs:
      outputs += tf.matmul(input, self.my_weight)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs
    
  def get_config(self):
    config = super(DensePermutation, self).get_config()
    config.update({
        "units": self.units,
        "activation": self.activation,
    })
    return config
  
def create_model():
  """Creates 12-tower-input model that outputs
  probability of winning a round."""
  tower_layer = Dense(32, activation="relu")
  total_tower_layer = DensePermutation(32, activation="relu")

  tower_inputs = Input(shape =(12,81), name = "Towers") 
  other_inputs = Input(shape = (46,), name = "Other")

  total_inputs = [other_inputs, tower_inputs]

  tower_layers = []
  for i in range(12):
    tower_layers.append(tower_layer(tower_inputs[:,i]))

  
  x = Dense(32, activation="relu")(Concatenate()([other_inputs, total_tower_layer(tower_layers)]))
  x = Dense(16, activation="relu")(x)
  y = Dense(1, activation = "sigmoid", name = 'odds_output')(x)

  model = Model(inputs=total_inputs, outputs=y,name='round_predictor')
  model.compile(optimizer='adam', metrics = ["binary_accuracy", tf.keras.metrics.Recall()], loss =tf.keras.losses.BinaryCrossentropy())
  return model