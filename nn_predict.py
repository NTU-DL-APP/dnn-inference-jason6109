import numpy as np

def relu(x):
    return np.maximum(0, np.array(x))

def softmax(x):
    shifted_x = x - np.max(x, axis=-1, keepdims=True)  # 確保處理多維輸入
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, img):
    layers = model_arch['config']['layers']
    x = img.reshape(1, -1)  # 確保輸入為二維陣列
    
    # 跳過 InputLayer 和 Flatten 層
    dense_layers = [layer for layer in layers if layer['class_name'] == 'Dense']
    
    for i, layer in enumerate(dense_layers):
        w = weights[f'arr_{i*2}']
        b = weights[f'arr_{i*2+1}']
        x = np.dot(x, w) + b
        
        activation = layer['config']['activation']
        if activation == 'relu':
            x = relu(x)
        elif activation == 'softmax':
            x = softmax(x)
            
    return x
