# Handwritten Digit Classification on the MNIST Dataset
*A Neural Network Approach Using TensorFlow / Keras*

## 1. Introduction

The goal of this assignment is to build a simple neural network that classifies handwritten digits (0-9) from the MNIST dataset using TensorFlow/Keras. The task covers the full pipeline expected of a supervised deep learning project: loading and exploring the data, designing and training a classifier, evaluating it on unseen data, visualizing learning behaviour, testing it on individual images, and running a controlled experiment to see how a single design choice affects performance.

MNIST consists of 70,000 grayscale images of handwritten digits, each 28x28 pixels: 60,000 for training and 10,000 held out for testing. It is a standard benchmark because it is small enough to train quickly on a CPU while still requiring a real model — not just a lookup table — to classify accurately.

## 2. Data Loading and Exploration

The dataset was loaded directly through `keras.datasets.mnist`, which returns NumPy arrays for the training and test images and labels:

- `x_train`: 60,000 images, shape `(60000, 28, 28)`, pixel values 0-255
- `y_train`: 60,000 integer labels, values 0-9
- `x_test`: 10,000 images, shape `(10000, 28, 28)`
- `y_test`: 10,000 integer labels, values 0-9

One example image was pulled from each of the ten classes to confirm the labels line up with what is actually drawn:

![One sample image per digit class](images/fig1_samples.png)

*Figure 1. One sample image per digit class (0-9) from the MNIST training set.*

Before training, two preprocessing steps were applied:

- Pixel values were scaled from the 0-255 range down to 0.0-1.0 by dividing by 255, which speeds up and stabilises gradient descent.
- Each 28x28 image was flattened into a 784-length vector (`x_train.reshape(x_train.shape[0], -1)`) since the network's first layer is a plain Dense layer, not a convolutional one.

Labels were one-hot encoded with `keras.utils.to_categorical(y_train, 10)` so that the model's 10-unit softmax output could be trained with `categorical_crossentropy`.

## 3. Model Design, Compilation, and Training

A fully-connected (dense) neural network was used as the baseline classifier:

- `Dense(128, activation='relu')` — first hidden layer, takes the 784-pixel input
- `Dense(128, activation='relu')` — second hidden layer
- `Dropout(0.25)` — randomly deactivates 25% of the units during training to reduce overfitting
- `Dense(10, activation='softmax')` — output layer producing a probability for each digit

The model was compiled with the Adam optimizer, `categorical_crossentropy` loss (appropriate for one-hot encoded multi-class labels), and accuracy as the tracked metric. It was trained for 10 epochs with a batch size of 512.

**Total trainable parameters: 118,282**

## 4. Evaluation on the Test Set

After training, the model was evaluated on the 10,000 held-out test images (data it never saw during training):

- Test loss: **0.0727**
- Test accuracy: **97.67%**

Predictions were compared against true labels for every test image, and a confusion matrix was built to see exactly where the model gets confused:

![Confusion matrix](images/fig2_confusion_matrix.png)

*Figure 2. Confusion matrix on the 10,000-image test set. Rows are true labels, columns are predicted labels.*

Out of 10,000 test images, 233 were misclassified (97.67% accuracy). The diagonal dominates the matrix, as expected; the small amount of off-diagonal mass is concentrated in digit pairs that are visually similar when handwritten, such as 4/9, 3/5, and 7/2.

## 5. Training and Validation Curves

Accuracy and loss were tracked at the end of every epoch to visualize how learning progressed:

![Training accuracy and loss curves](images/fig3_training_curves.png)

*Figure 3. Training accuracy (left) and training loss (right) over 10 epochs, shown for both the baseline (128 neurons/layer) and the experiment model (256 neurons/layer) described in Section 7.*

Both curves show steady, expected learning behaviour: accuracy rises and loss falls smoothly across all 10 epochs with no instability, and neither model shows signs of divergence. The baseline model's training accuracy climbs from about 82.6% after epoch 1 to 98.5% by epoch 10, while training loss drops from 0.60 to 0.05 over the same period.

## 6. Testing on 5 Individual Images

Five images were sampled from the test set and passed through the trained model one at a time, with the predicted label compared directly against the true label:

![Five sample predictions](images/fig4_five_predictions.png)

*Figure 4. Five individual test images with the model's predicted label and the true label. Green titles indicate a correct prediction.*

## 7. Experiment: Changing the Number of Neurons per Hidden Layer

To measure the effect of a single design choice, a second model was trained that is identical to the baseline in every respect except one: the number of neurons in each of the two hidden layers was increased from 128 to 256. Architecture (2 hidden layers), activation (relu), dropout rate (0.25), optimizer (Adam), loss function, batch size (512), and epoch count (10) were all held constant, so any change in performance can be attributed to this one change alone.

### 7.1 Results

| Metric | Baseline (128 neurons/layer) | Experiment (256 neurons/layer) |
|---|---|---|
| Hidden layers | 2 (Dense 128, Dense 128) | 2 (Dense 256, Dense 256) — unchanged count |
| Dropout | 0.25 after 2nd hidden layer | 0.25 after 2nd hidden layer — unchanged |
| Trainable parameters | 118,282 | 269,322 |
| Optimizer / Loss | Adam / categorical_crossentropy | Adam / categorical_crossentropy — unchanged |
| Batch size / Epochs | 512 / 10 | 512 / 10 — unchanged |
| Final training accuracy | 98.50% | 99.20% |
| Final training loss | 0.0497 | 0.0264 |
| Test accuracy | 97.67% | 98.12% |
| Test loss | 0.0727 | 0.0654 |
| Misclassified images (of 10,000) | 233 | 188 |

![Before vs after comparison](images/fig5_before_after_bars.png)

*Figure 5. Test accuracy and test loss before (128 neurons/layer) and after (256 neurons/layer) the experiment.*

### 7.2 Discussion

Doubling the width of each hidden layer improved test accuracy from 97.67% to 98.12% (a gain of 0.45 percentage points) and reduced test loss from 0.0727 to 0.0654. The number of misclassified test images fell from 233 to 188, roughly a 19% reduction in errors.

This came at a clear cost: the parameter count more than doubled, from 118,282 to 269,322, and each epoch took noticeably longer to train (roughly 10-13 ms/step versus 5-12 ms/step for the smaller model). The wider model also reached a higher final training accuracy (99.20% vs 98.50%), showing it has more capacity to fit the training data — with dropout still active at the same 0.25 rate in both models, this extra capacity translated into better test performance rather than pronounced overfitting, since the test accuracy improved alongside the training accuracy rather than pulling away from it.

Overall, more neurons per layer gave a modest but consistent improvement on this dataset. The gain is smaller than what would typically be expected from architectural changes such as adding convolutional layers, which better exploit the 2D spatial structure of image data — a natural next step beyond this experiment.

## 8. Conclusion

A simple two-hidden-layer dense network with dropout reached 97.67% test accuracy on MNIST after just 10 epochs of training, and correctly classified all five individually tested sample images. Increasing the hidden layer width from 128 to 256 neurons — the one variable changed in the experiment — raised test accuracy to 98.12% while keeping every other setting fixed, confirming that model capacity, even within a simple fully-connected architecture, has a measurable effect on classification performance for this task.
