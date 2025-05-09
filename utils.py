import numpy as np
import tensorflow as tf

def data_preprocessing(x_train, y_train, x_test):
    ''' Preprocess the 2D numpy arrays into tensors for CNN model.
    Input: numpy arrays of shape (n_samples, n_features)
    Output: 4D tensors of shape (n_samples, n_rows, n_cols, n_channels)
    '''
    n_shape = int(np.sqrt(x_train.shape[1]))

    x_train_img = x_train.reshape(-1, n_shape, n_shape)
    x_train_tensor = x_train_img.reshape(
        x_train_img.shape[0], x_train_img.shape[1], x_train_img.shape[2], 1)
    y_train_tensor = tf.one_hot(y_train.astype(np.int32), depth=10)
    x_test_tensor = x_test.reshape(x_test.shape[0], n_shape, n_shape, 1)

    return x_train_img, x_train_tensor, y_train_tensor, x_test_tensor


class accuracy_callback(tf.keras.callbacks.Callback):
  ''' Callback for CNN model to stop training when accuracy reaches 99% '''

  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('val_acc') > 0.996):
      print(f"\nReached 99% accuracy at epoch {epoch}. Stop training.")
      self.model.stop_training = True
