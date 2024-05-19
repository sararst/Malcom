import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler

class MLP:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.initial_learning_rate = 0.01
        self.decay_steps = 1000
        self.decay_rate = 0.95
        self.weight_decay = 0.01

    def lr_scheduler(self, epoch, lr):
        if epoch % self.decay_steps == 0 and epoch:
            return lr * self.decay_rate
        return lr

    def compile(self):
        optimizer = Adam(learning_rate=self.initial_learning_rate, decay=self.weight_decay)
        self.model.compile(optimizer=optimizer,
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    def fit(self, x, y, epochs, batch_size):
        lr_callback = LearningRateScheduler(self.lr_scheduler)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=lr_callback)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)