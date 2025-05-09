# CNN TensorFlow Project

This folder contains a simple example of training an MLP for classification tasks using TensorFlow.

## Contents

- `cnn_model.py`: Defines the `MLP` class with methods for compiling, fitting, predicting, and evaluating the model.
- `utils.py`: Provides data preprocessing functions and a custom `accuracy_callback`.

## Usage

1. **Data Preprocessing**  
   Use `data_preprocessing(x_train, y_train, x_test)` from [utils.py](utils.py) to convert 2D numpy arrays into the appropriate 4D format:
   ```python
   from utils import data_preprocessing
   x_train_img, x_train_tensor, y_train_tensor, x_test_tensor = data_preprocessing(x_train, y_train, x_test)
   ```

2. **Building and Compiling the Model**  
   Create an `MLP` instance and compile it:
   ```python
   from cnn_model import MLP

   model = MLP(
       batch_size=32,
       num_classes=10,
       epochs=10,
       learning_rate=0.001,
       callbacks=[]
   ).compile_model()
   ```

3. **Training the Model**  
   Fit the model on training data:
   ```python
   history = model.fit(x_train_tensor, y_train_tensor)
   ```

4. **Evaluating the Model**  
   Evaluate using test data:
   ```python
   acc, loss = model.evaluate(x_test_tensor, y_test)
   print("Accuracy:", acc)
   print("Loss:", loss)
   ```

5. **Predicting**  
   Generate predictions on new data:
   ```python
   y_pred = model.get_predictions(x_test_tensor)
   ```

6. **Custom Callback**  
   The `accuracy_callback` in [utils.py](utils.py) stops training when validation accuracy exceeds 99%.

## Requirements
- Python 3.x
- NumPy
- TensorFlow 2.x

## License
This code is provided for educational purposes.