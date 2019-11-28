import tensorflow as tf
import os

model_save_path = './checkpoint/mnist.tf'
load_pretrain_model = False

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if load_pretrain_model:
    print('----------load the model----------')
    model.load_weights(model_save_path)



for i in range(50):
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
    model.save_weights(model_save_path,save_format = 'tf')
model.summary()
