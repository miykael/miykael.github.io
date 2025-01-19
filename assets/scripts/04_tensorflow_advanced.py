import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import ParameterGrid
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory for figures if it doesn't exist
plot_dir = Path("../assets/ex_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# 1. Dataset Preparation
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]
df = pd.read_csv(url, header=None, names=columns)

# Convert categorical data to numerical
df = pd.get_dummies(df)
print(f"Shape of dataset: {df.shape}")

# Split dataset into train and test set
df_tr = df.sample(frac=0.8, random_state=0)
df_te = df.drop(df_tr.index)

# Separate target from features and convert to float32
x_tr = np.asarray(df_tr.drop(columns=["Rings"])).astype("float32")
x_te = np.asarray(df_te.drop(columns=["Rings"])).astype("float32")
y_tr = np.asarray(df_tr["Rings"]).astype("float32")
y_te = np.asarray(df_te["Rings"]).astype("float32")

print(f"Size of training and test set: {df_tr.shape} | {df_te.shape}")

# Normalize data with a keras layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_tr)

print(f"Mean parameters:\n{normalizer.adapt_mean.numpy()}\n")
print(f"Variance parameters:\n{normalizer.adapt_variance.numpy()}")


# 2. Model Creation

# Create layers and connect them with functional API
input_layer = keras.Input(shape=(x_tr.shape[1],))

# Normalize inputs using our pre-trained normalization layer
x = normalizer(input_layer)

# Build hidden layers with explicit connections
x = layers.Dense(8)(x)
x = layers.BatchNormalization()(x)  # Stabilizes training
x = layers.ReLU()(x)  # Non-linear activation
x = layers.Dropout(0.5)(x)  # Prevents overfitting

# Second dense layer with similar structure but fewer neurons
x = layers.Dense(4)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.5)(x)

# Output layer for regression (no activation function)
output_layer = layers.Dense(1)(x)

# Create model by specifying inputs and outputs
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Check model size
model.summary(show_trainable=False)


# Create and compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(name="MSE"),
    metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
)

# Save best performing model (based on validation loss) in checkpoint
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="model_backup",
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)

# Store training history in csv file
history_logger = keras.callbacks.CSVLogger(
    "history_log.csv", separator=",", append=False
)

# Reduce learning rate on plateau
reduce_lr_on_plateau = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=1e-5, verbose=0
    ),
)

# Use early stopping to stop learning once it doesn't improve anymore
early_stopping = (
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
)

# 3. Training
history = model.fit(
    x=x_tr,
    y=y_tr,
    validation_split=0.2,
    shuffle=True,
    batch_size=64,
    epochs=200,
    verbose=1,
    callbacks=[
        model_checkpoint_callback,
        history_logger,
        reduce_lr_on_plateau,
        early_stopping,
    ],
)


def plot_history(
    history_file="history_log.csv", title="", filename="output_filename.png"
):
    # Load training history from CSV
    history_log = pd.read_csv(history_file)

    # Create subplots for loss and MAE metrics
    _, axs = plt.subplots(1, 2, figsize=(15, 4))

    # Plot loss metrics
    history_log.iloc[:, history_log.columns.str.contains("loss")].plot(
        title="Loss during training", ax=axs[0]
    )
    # Plot MAE metrics
    history_log.iloc[:, history_log.columns.str.contains("MAE")].plot(
        title="MAE during training", ax=axs[1]
    )

    # Configure axis labels and scales
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("MAE")
    for idx in range(len(axs)):
        axs[idx].set_xlabel("Epoch [#]")
        axs[idx].set_yscale("log")  # Use log scale for better visualization of changes

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(
        plot_dir / filename,
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()


# Plot training history
plot_history(
    history_file="history_log.csv",
    title="Training overview",
    filename="04_tensorflow_training_history.png",
)

# 4. Inference
model_best = keras.models.load_model("model_backup")

# Evaluate best model on test set
train_results = model_best.evaluate(x_tr, y_tr, verbose=0)
test_results = model_best.evaluate(x_te, y_te, verbose=0)
print("Train - Loss: {:.3f} | MAE: {:.3f}".format(*train_results))
print("Test  - Loss: {:.3f} | MAE: {:.3f}".format(*test_results))

# Print model summary
model.summary(show_trainable=False)


# 5. Architecture fine-tuning
def build_and_compile_model(
    hidden=[8, 4],  # List defining sizes of hidden layers
    activation="relu",  # Activation function for hidden layers
    use_batch=True,  # Whether to use batch normalization
    dropout_rate=0,  # Dropout rate for regularization
    learning_rate=0.001,  # Initial learning rate
    optimizers="adam",  # Choice of optimizer
    kernel_init="he_normal",  # Weight initialization strategy
    kernel_regularizer=None,  # Weight regularization method
):
    # Create input layer
    input_layer = keras.Input(shape=(x_tr.shape[1],))

    # Normalize input data
    x = normalizer(input_layer)

    # Build hidden layers
    for idx, h in enumerate(hidden):
        # Add batch normalization if requested
        if use_batch:
            x = layers.BatchNormalization()(x)

        # Add dense layer with specific parameters
        x = layers.Dense(
            h,
            activation=activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_regularizer,
        )(x)

        # Add dropout layer if requested
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    # Add output layer (no activation for regression)
    output_layer = layers.Dense(1)(x)

    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Configure optimizer based on selection
    if optimizers == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizers == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizers == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)

    # Compile model with loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(name="MSE"),
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
    )
    return model


# Use function to create model and report summary overview
model = build_and_compile_model()
model.summary()

# Define exploration grid
hidden = [[8], [8, 4], [8, 16, 8]]
activation = ["relu", "elu", "selu", "tanh", "sigmoid"]
use_batch = [False, True]
dropout_rate = [0, 0.5]
learning_rate = [1e-3, 1e-4]
optimizers = ["adam", "rmsprop", "sgd"]
kernel_init = [
    "he_normal",
    "he_uniform",
    "glorot_normal",
    "glorot_uniform",
    "uniform",
    "normal",
]
kernel_regularizer = [None, "l1", "l2", "l1_l2"]
batch_sizes = [32, 128]

# Create parameter grid
param_grid = dict(
    hidden=hidden,  # Network architectures from simple to complex
    activation=activation,  # Different non-linearities for different data patterns
    use_batch=use_batch,  # Batch normalization for training stability
    dropout_rate=dropout_rate,  # Regularization to prevent overfitting
    learning_rate=learning_rate,  # Controls step size during optimization
    optimizers=optimizers,  # Different optimization strategies
    kernel_init=kernel_init,  # Weight initialization methods
    kernel_regularizer=kernel_regularizer,  # Weight penalties to prevent overfitting
    batch_size=batch_sizes,  # Training batch sizes for different memory/speed tradeoffs
)

grid = ParameterGrid(param_grid)
print(f"Exploring {len(grid)} network architectures.")

# Number of grid points to explore
nth = 5

# Establish grid indecies, shuffle them and keep the first N-th entries
grid_idx = np.arange(len(grid))
np.random.shuffle(grid_idx)
grid_idx = grid_idx[:nth]

# Loop through grid points
for gixd in grid_idx:
    # Select grid point
    g = grid[gixd]
    print(f"Exploring: {g}")

    # Build and compile model
    model = build_and_compile_model(
        hidden=g["hidden"],
        activation=g["activation"],
        use_batch=g["use_batch"],
        dropout_rate=g["dropout_rate"],
        learning_rate=g["learning_rate"],
        optimizers=g["optimizers"],
        kernel_init=g["kernel_init"],
        kernel_regularizer=g["kernel_regularizer"],
    )

    # Save best performing model (based on validation loss) in checkpoint
    backup_path = f"model_backup_{gixd}"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=backup_path,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    # Store training history in csv file
    history_file = f"history_log_{gixd}.csv"
    history_logger = keras.callbacks.CSVLogger(
        history_file, separator=",", append=False
    )

    # Setup callbacks
    callbacks = [
        model_checkpoint_callback,
        history_logger,
        reduce_lr_on_plateau,
        early_stopping,
    ]

    # Train model
    history = model.fit(
        x=x_tr,
        y=y_tr,
        validation_split=0.2,
        shuffle=True,
        batch_size=g["batch_size"],
        epochs=200,
        callbacks=callbacks,
        verbose=0,
    )

    # Load best model
    model_best = keras.models.load_model(backup_path)

    # Evaluate best model on test set
    train_results = model_best.evaluate(x_tr, y_tr, verbose=0)
    test_results = model_best.evaluate(x_te, y_te, verbose=0)
    print("\tTrain - Loss: {:.3f} | MAE: {:.3f}".format(*train_results))
    print("\tTest  - Loss: {:.3f} | MAE: {:.3f}".format(*test_results))
    print(f"\tModel Parameters: {model.count_params()}\n\n")

# 6. Fine-tuning investigation
results = []

# Loop through grid points
for gixd in grid_idx:
    # Select grid point
    g = grid[gixd]

    # Restore best model
    backup_path = f"model_backup_{gixd}"
    model_best = keras.models.load_model(backup_path)

    # Evaluate best model on test set
    train_results = model_best.evaluate(x_tr, y_tr, verbose=0)
    test_results = model_best.evaluate(x_te, y_te, verbose=0)

    # Store information in table
    df_score = pd.Series(g)
    df_score["loss_tr"] = train_results[0]
    df_score["MAE_tr"] = train_results[1]
    df_score["loss_te"] = test_results[0]
    df_score["MAE_te"] = test_results[1]
    df_score["idx"] = gixd

    results.append(df_score.to_frame().T)

results = pd.concat(results).reset_index(drop=True).sort_values("loss_te")

# Get idx of best model
gixd_best = results.iloc[0, -1]

# Load history file of best training
history_file = f"history_log_{gixd_best}.csv"

# Plot training curves
plot_history(
    history_file=history_file,
    title="Training overview of best model",
    filename="04_tensorflow_architecture_comparison.png",
)
