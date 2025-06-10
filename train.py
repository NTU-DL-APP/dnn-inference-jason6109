import tensorflow as tf
import numpy as np
import os

# 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型（只能有 Dense 層）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯與訓練
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=256)

# 輸出檔案
os.makedirs('./model', exist_ok=True)
with open('./model/fashion_mnist.json', 'w') as f:
    f.write(model.to_json())
np.savez('./model/fashion_mnist.npz', *model.get_weights())
