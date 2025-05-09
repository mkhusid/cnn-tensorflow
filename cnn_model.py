import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Input, BatchNormalization, Dropout


class MLP:
    def __init__(self, batch_size: int, num_classes: int,
                 epochs: int, learning_rate: float, callbacks: tf.keras.callbacks.Callback):
        ''' Initialize the model with the hyperparameters '''
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.callbacks = callbacks

    def _set_layers(self, d_out=0.5, hdn_act='relu'):
        ''' Set the layers of the MLP model: input, hidden, output
        Args:
            d_out: dropout rate (0.3-0.5 is recommended)
            hdn_act: activation function for hidden layers
        '''
        def hidden_layer(neurons): return \
            (Dense(neurons, activation=hdn_act), BatchNormalization(), Dropout(d_out))

        self.layers = [
            # Flatten input: 28x28 images => 784 vector
            Input(shape=(28, 28, 1)),
            Flatten(),

            # Hidden layers: 1024->512->256->128 neurons
            *hidden_layer(1024),
            *hidden_layer(512),
            *hidden_layer(256),
            *hidden_layer(128),

            # Output layer: 10 classes
            Dense(10, activation='softmax')
        ]

    def compile_model(self):
        ''' Compile the model with the optimizer, loss function, and metrics '''
        self._set_layers()
        self.model: tf.keras.Model = Sequential(self.layers)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        return self

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        ''' Fit the model with the training data
        Args:
            x_train: training data
            y_train: training labels
        '''
        return self.model.fit(x_train, y_train,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              validation_split=0.1,
                              callbacks=self.callbacks)

    def get_predictions(self, x_test: np.ndarray):
      ''' Get the predictions from the model on the testing dataset '''
      y_pred = self.model.predict(x_test)
      return np.argmax(y_pred, axis=1)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        ''' Evaluate the model with the testing data '''
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print("Final Accuracy:", acc)
        print("Final Loss:", loss)
        return acc, loss
