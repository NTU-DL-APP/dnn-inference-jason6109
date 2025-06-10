import tensorflow as tf
import numpy as np
import os

# 1. 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 建立模型（只能用 Dense, activation='relu'/'softmax'）
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型
model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.1)

# 4. 評估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'測試集準確率：{test_acc:.4f}')

# 5. 匯出模型
os.makedirs('./model', exist_ok=True)
# (1) 匯出架構
with open('./model/fashion_mnist.json', 'w') as f:
    f.write(model.to_json())
# (2) 匯出權重
weights = model.get_weights()
weight_dict = {f'arr_{i}': w for i, w in enumerate(weights)}
np.savez('./model/fashion_mnist.npz', **weight_dict)
