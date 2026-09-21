# MNIST Handwritten Digit Classification Using Neural Network

## Project Overview

This project implements a simple neural network using **TensorFlow/Keras** to classify handwritten digits from **0 to 9** using the MNIST handwritten digit dataset.

The project covers the complete machine learning workflow, including dataset exploration, preprocessing, model development, training, evaluation, visualization, prediction, and model comparison.

## Objectives

The main objectives of this project are:

* Load and explore the MNIST dataset.
* Display sample handwritten digit images.
* Preprocess and normalize the image data.
* Design and train a neural network using TensorFlow/Keras.
* Evaluate the model using test accuracy.
* Visualize training and validation accuracy.
* Visualize training and validation loss.
* Test the model on five handwritten images.
* Compare actual and predicted digit labels.
* Perform an experiment by adding Dropout.
* Compare the baseline and experimental models.

## Dataset

The **MNIST dataset** contains handwritten grayscale digit images representing the numbers 0 through 9.

Each image has a resolution of **28 × 28 pixels**.

The dataset contains:

* **60,000 training images**
* **10,000 test images**
* **10 different classes**
* Classes: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9**

The dataset is loaded directly using TensorFlow/Keras:

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

## Data Preprocessing

The original pixel values range from 0 to 255.

The images are normalized to a range between 0 and 1:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

Normalization helps the neural network train more effectively.

## Neural Network Model

The baseline neural network consists of:

* Flatten layer
* Dense layer with 128 neurons and ReLU activation
* Dense layer with 64 neurons and ReLU activation
* Output Dense layer with 10 neurons and Softmax activation

The output layer provides probabilities for the ten possible digit classes.

## Model Compilation

The model uses:

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Crossentropy

**Evaluation Metric:** Accuracy

Example:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Model Training

The model is trained for 10 epochs with a batch size of 32.

A validation split of 20% is used during training to monitor the model's performance on unseen validation data.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
```

## Model Evaluation

After training, the model is evaluated using the MNIST test dataset.

The test loss and test accuracy are calculated using:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)
```

The exact accuracy obtained can vary slightly depending on the training run.

## Visualization

The project visualizes:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

These graphs help analyze the learning behavior of the neural network during training.

## Five Image Prediction Test

Five handwritten digit images from the MNIST test dataset are selected and passed through the trained model.

For each image, the project displays:

* Actual digit
* Predicted digit
* Whether the prediction is correct or incorrect

The prediction is obtained using:

```python
prediction = model.predict(sample_images)

predicted_labels = np.argmax(
    prediction,
    axis=1
)
```

## Experiment: Dropout

An additional experiment is performed by adding **Dropout layers** to the neural network.

The experimental model uses:

```python
Dropout(0.2)
```

after the hidden Dense layers.

Dropout randomly deactivates a portion of neurons during training and can help reduce overfitting.

The baseline and Dropout models are compared using their:

* Test accuracy
* Test loss
* Validation accuracy

The exact experimental results are generated when the notebook is executed.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## How to Run

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the MNIST notebook and run all cells from beginning to end.

The MNIST dataset will be downloaded automatically through TensorFlow/Keras when required.

## Results

The notebook generates the following results:

* MNIST sample image visualization
* Training accuracy graph
* Validation accuracy graph
* Training loss graph
* Validation loss graph
* Five-image actual vs predicted comparison
* Baseline model test accuracy
* Dropout model test accuracy
* Validation accuracy comparison

The numerical results should be taken directly from the executed notebook.

## Conclusion

This project demonstrates how a neural network can be used to classify handwritten digits using the MNIST dataset.

The project covers the complete workflow from loading and preprocessing the dataset to training, evaluating, visualizing, and testing the neural network.

The Dropout experiment further demonstrates how modifying the neural network architecture can affect model performance and help address overfitting.

## Author

**Vardhan R A.**

### Project

**Handwritten Digit Classification Using Neural Network**

### Framework

**TensorFlow / Keras**
