import tensorflow as tf
from tensorflow.keras.utils import plot_model

model = tf.keras.models.load_model('IMDBLTST2.h5')

plot_model(model, to_file='tfmodel.png', show_shapes=True, show_layer_names=True)

#other ways to visualize model
