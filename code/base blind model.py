from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def selective_amnesia(x_train, y_train, x_test, y_test, epochs=1000, recovery_epochs=10, 
                       forget_threshold=0.1, recover_threshold=0.01):
  """
  Implementing selective amnesia with adaptive thresholds.

  Args:
      x_train: Training data (images).
      y_train: Training labels.
      x_test: Testing data (images).
      y_test: Testing labels.
      epochs: Number of epochs for initial training with wrong labels.
      recovery_epochs: Number of epochs for recovery training.
      forget_threshold: Minimum accuracy drop to trigger forgetting.
      recover_threshold: Minimum accuracy gain to stop recovery training.


  """

  model = Sequential([
      Dense(512, activation="relu", input_shape=(x_train.shape[1],)),
      Dense(10, activation="softmax")
  ])
  model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

 
  initial_acc = model.evaluate(x_test, y_test)[1]

  # Training with wrong labels (amnesia phase)
  wrong_labels = np.random.randint(0, 10, size=(y_train.shape[0],))  # Generate random wrong labels
  wrong_labels[wrong_labels == y_train] = (wrong_labels[wrong_labels == y_train] + 1) % 10  # Ensure different labels

  model.fit(x_train, wrong_labels, epochs=epochs)  # Train with wrong labels

  # Evaluate and check for forgetting condition
  acc = model.evaluate(x_test, y_test)[1]
  delta_acc = initial_acc - acc
  if delta_acc > forget_threshold:
    return model, test_acc  # Exit function if accuracy drop is significant

  # Forgetting phase (reset model)
  model.set_weights([weights.copy() for weights in model.get_weights()])

  # Recovery training with adaptive thresholds
  prev_acc = 0  # Initialize previous accuracy
  for epoch in range(recovery_epochs):
    model.train_on_batch(x_train, y_train)
    acc = model.evaluate(x_test, y_test)[1]
    delta_acc = acc - prev_acc
    prev_acc = acc  # Update previous accuracy

    # Update forgetting threshold
    forget_threshold -= delta_acc * 0.1  # Adjust based on forgetting rate (example)

    if delta_acc > recover_threshold:
      break

    # Update recovery threshold
    recover_threshold += delta_acc * 0.1  # Adjust based on recovery rate (example)

  # Final test accuracy
  test_acc = model.evaluate(x_test, y_test)[1]
  return model, test_acc


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

model, test_acc = selective_amnesia(x_train, y_train, x_test, y_test)

