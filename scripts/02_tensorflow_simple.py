import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory for figures if it doesn't exist
plot_dir = Path("../assets/ex_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# 1. Load and Prepare Dataset
num_classes = 10  # One class for each digit (0-9)
input_shape = (28, 28, 1)  # Height, width, and channels (1 for grayscale)

# Load dataset, already pre-split into train and test set
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale pixel values to range [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension required by Conv2D layers
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# 2. Create Neural Network Model - Layer-by-Layer Sequential API
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3)),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3)),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(32),
        layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(num_classes),
        layers.Softmax(),
    ]
)

# Print model summary
model.summary()

# 3. Train TensorFlow model
batch_size = 128
epochs = 10

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the model
history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

# 4. Model investigation
# Store history in a dataframe
df_history = pd.DataFrame(history.history)

# Visualize training history
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
df_history.iloc[:, df_history.columns.str.contains("loss")].plot(
    title="Loss during training", ax=axs[0]
)
df_history.iloc[:, df_history.columns.str.contains("accuracy")].plot(
    title="Accuracy during training", ax=axs[1]
)
axs[0].set_xlabel("Epoch [#]")
axs[1].set_xlabel("Epoch [#]")
axs[0].set_ylabel("Loss")
axs[1].set_ylabel("Accuracy")
plt.savefig(
    plot_dir / "02_tensorflow_training_history.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# Evaluate model on test set
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss:     {score[0]:.3f}")
print(f"Test accuracy: {score[1] * 100:.2f}%")

# Get predictions
y_pred = model.predict(x_test, verbose=0)
print("Prediction shape:", y_pred.shape)

# Transform class probabilities to prediction labels
predictions = np.argmax(y_pred, 1)

# Create and visualize confusion matrix
cm = tf.math.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, square=True, annot=True, fmt="d", cbar=False)
plt.title("Confusion matrix")
plt.savefig(
    plot_dir / "02_tensorflow_confusion_matrix.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# Visualize first convolutional layer kernels
conv_layer = model.layers[0]
weights = conv_layer.weights[0].numpy()

# Visualize the 32 kernels from the first convolutional layer
fig, axs = plt.subplots(4, 8, figsize=(10, 5))
axs = np.ravel(axs)

for idx, ax in enumerate(axs):
    ax.set_title(f"Kernel {idx}")
    ax.imshow(weights[..., idx], cmap="binary")
    ax.axis("off")
plt.tight_layout()
plt.savefig(
    plot_dir / "02_tensorflow_conv_kernels.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# Clear memory after training
gc.collect()
keras.backend.clear_session()
