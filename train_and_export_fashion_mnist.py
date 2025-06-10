import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# 1. 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# 4. 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'測試集準確率：{test_acc:.4f}')

# 5. 輸出檔案
os.makedirs('./model', exist_ok=True)
# (1) 儲存完整模型（可選）
model.save('./model/fashion_mnist.h5')
# (2) 儲存模型架構
with open('./model/fashion_mnist.json', 'w') as f:
    f.write(model.to_json())
# (3) 儲存模型權重
weights = model.get_weights()
weight_dict = {f'weight_{i}': w for i, w in enumerate(weights)}
np.savez('./model/fashion_mnist.npz', **weight_dict)

print('模型檔案已儲存至 ./model 資料夾！')
