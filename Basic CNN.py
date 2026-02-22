import numpy as np
import matplotlib.pyplot as plt

class Conv2D:
    def __init__(self, num_filters, in_channels, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.filters = np.random.randn(num_filters, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros(self.num_filters)
    def forward(self, input_image):
        (C_in, H_in, W_in) = input_image.shape

        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1

        output = np.zeros((self.num_filters, H_out, W_out))

        for f in range(self.num_filters):
            for h in range(H_out):
                for w in range(W_out):
                    patch = input_image[:, h : h + self.kernel_size, w : w + self.kernel_size]

                    conv_sum = np.sum(patch * self.filters[f])
                    output[f, h, w] = conv_sum + self.bias[f]
        return output

class Relu:
    def forward(self, input_data):
        return np.maximum(0, input_data)

class MaxPool:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_map):
        (C_in, H_in, W_in) = input_map.shape
        H_out = int((H_in - self.pool_size) / self.stride) + 1
        W_out = int((W_in - self.pool_size) / self.stride) + 1
        output = np.zeros((C_in, H_out, W_out))

        for c in range(C_in):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    patch = input_map[c, h_start: h_start + self.pool_size, w_start: w_start + self.pool_size]
                    output[c, h, w] = np.max(patch)
        return output

class Flatten:
    def forward(self, input_data):
        return input_data.ravel()

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)

    def forward(self, input_vector):
        return np.dot(input_vector, self.weights) + self.bias

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


image = np.zeros((28, 28))

image = np.zeros((28, 28))

# === EARS (Realistic triangular with inner pink) ===
# Left ear outer
image[3:8, 4:10] = 0.75   # Left ear base
image[3:5, 5:9] = 0.80    # Left ear top
image[4:6, 6:8] = 0.85    # Left ear highlight
# Left ear inner (pink)
image[5:7, 6:8] = 0.95    # Inner ear pink
image[6, 7] = 0.90        # Inner ear shadow

# Right ear outer
image[3:8, 18:24] = 0.75  # Right ear base
image[3:5, 19:23] = 0.80  # Right ear top
image[4:6, 20:22] = 0.85  # Right ear highlight
# Right ear inner (pink)
image[5:7, 20:22] = 0.95  # Inner ear pink
image[6, 20] = 0.90       # Inner ear shadow

# Ear tips (darker fur)
image[3, 6:8] = 0.65      # Left ear tip
image[3, 20:22] = 0.65    # Right ear tip

# === HEAD SHAPE (Fluffy, rounded) ===
image[7:23, 6:22] = 0.70   # Base head
image[9:21, 5:23] = 0.75   # Wider section
image[11:19, 4:24] = 0.78  # Widest (cheeks)
image[13:17, 3:25] = 0.80  # Maximum width

# Forehead fur texture
image[8:11, 10:18] = 0.82
image[9, 12:16] = 0.85

# === EYES (Large, expressive anime-style) ===
# Left eye
image[10:15, 7:13] = 0.98  # Eye white
image[11:14, 8:12] = 0.95  # Eye ball
image[12:14, 9:11] = 0.25  # Iris (colored)
image[12:13, 9:11] = 0.15  # Pupil top
image[13, 9:11] = 0.08     # Pupil bottom (darker)
image[11, 9:10] = 1.0      # Eye shine top
image[13, 10:11] = 0.92    # Eye shine bottom
image[10, 7:13] = 0.60     # Upper eyelid
image[15, 8:12] = 0.55     # Lower eyelid shadow

# Right eye
image[10:15, 15:21] = 0.98 # Eye white
image[11:14, 16:20] = 0.95 # Eye ball
image[12:14, 17:19] = 0.25 # Iris (colored)
image[12:13, 17:19] = 0.15 # Pupil top
image[13, 17:19] = 0.08    # Pupil bottom (darker)
image[11, 18:19] = 1.0     # Eye shine top
image[13, 17:18] = 0.92    # Eye shine bottom
image[10, 15:21] = 0.60    # Upper eyelid
image[15, 16:20] = 0.55    # Lower eyelid shadow

# Eye outlines (darker fur around eyes)
image[9, 7:13] = 0.50      # Left eye top outline
image[15, 7:13] = 0.50     # Left eye bottom outline
image[9, 15:21] = 0.50     # Right eye top outline
image[15, 15:21] = 0.50    # Right eye bottom outline

# === NOSE (Pink, triangular, 3D) ===
image[16:18, 13:15] = 0.95 # Nose main (pink)
image[16, 13:15] = 0.98    # Nose top highlight
image[17, 13:15] = 0.90    # Nose bottom
image[17, 13] = 0.85       # Left nostril
image[17, 14] = 0.85       # Right nostril
image[15, 13:15] = 0.65    # Nose bridge

# === MUZZLE (White/light fur area) ===
image[16:21, 10:18] = 0.88 # Muzzle base
image[17:20, 11:17] = 0.92 # Muzzle center (lighter)
image[18:19, 12:16] = 0.95 # Muzzle highlight

# === MOUTH (Cute cat mouth) ===
image[18, 14] = 0.40       # Center line from nose
image[19, 13] = 0.45       # Left mouth curve
image[19, 14] = 0.45       # Center
image[19, 15] = 0.45       # Right mouth curve
image[20, 12] = 0.50       # Left smile
image[20, 16] = 0.50       # Right smile

# === WHISKERS (Multiple, detailed) ===
# Left whiskers
image[17, 3:10] = 0.30     # Top left whisker
image[18, 2:10] = 0.30     # Middle left whisker
image[19, 3:10] = 0.30     # Bottom left whisker
image[16, 4:9] = 0.35      # Upper left whisker

# Right whiskers
image[17, 18:25] = 0.30    # Top right whisker
image[18, 18:26] = 0.30    # Middle right whisker
image[19, 18:25] = 0.30    # Bottom right whisker
image[16, 19:24] = 0.35    # Upper right whisker

# Whisker dots (where whiskers grow from)
image[17, 10] = 0.50       # Left whisker dot
image[18, 10] = 0.50
image[17, 17] = 0.50       # Right whisker dot
image[18, 17] = 0.50

# === CHEEKS (Fluffy fur) ===
image[15:19, 5:8] = 0.82   # Left cheek fluff
image[16:18, 4:6] = 0.85   # Left cheek highlight
image[15:19, 20:23] = 0.82 # Right cheek fluff
image[16:18, 22:24] = 0.85 # Right cheek highlight

# === CHIN (Rounded, white) ===
image[20:23, 11:17] = 0.85 # Chin base
image[21, 12:16] = 0.90    # Chin center
image[22, 13:15] = 0.88    # Chin highlight


# Forehead stripes (tabby pattern)
image[9, 11:13] = 0.60
image[9, 15:17] = 0.60
image[10, 10:12] = 0.65
image[10, 16:18] = 0.65
image[8, 12:16] = 0.65

# Cheek stripes
image[14, 6:8] = 0.60
image[14, 20:22] = 0.60


# Face contour shadows
image[11:15, 6] = 0.65
image[11:15, 21] = 0.65
image[18:21, 8:10] = 0.75
image[18:21, 18:20] = 0.75


image[22:24, 11:17] = 0.70
image[23, 12:16] = 0.65

image[12:14, 13:15] = 0.68

image[12, 10] = 0.90
image[12, 17] = 0.90
image[10, 13:15] = 0.85
image[14, 13:15] = 0.70

# Add realistic skin texture and variation
fur_texture = np.random.rand(28, 28) * 0.08
image = image + fur_texture
print(image)
for row in range(28):
    row_string = " ".join(["█" if val >= 0.2 else "0" for val in image[row, :]])
    print(row_string)

print("Generated Human Face Input (28x28)")
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title("Human")
plt.show()
input_image = image.reshape(1, 28, 28)
print(f"\nSTART Input Shape: {input_image.shape}")

conv_layer = Conv2D(in_channels=1, num_filters=4, kernel_size=3)
conv_output = conv_layer.forward(input_image)
print(f"   Output Shape: {conv_output.shape}")
relu_layer = Relu()
relu_output = relu_layer.forward(conv_output)
print(f"   Output Shape: {relu_output.shape}")
maxpool_layer = MaxPool(pool_size=2, stride=2)
maxpool_output = maxpool_layer.forward(relu_output)
print(f"   Output Shape: {maxpool_output.shape}")
flatten_layer = Flatten()
flatten_output = flatten_layer.forward(maxpool_output)
print(f"   Output Shape: {flatten_output.shape}")
dense_layer = Dense(input_size=676, output_size=10)
dense_output = dense_layer.forward(flatten_output)
print(f"   Output Shape: {dense_output.shape}")
print("\n6. Softmax (Final Probabilities)")

probabilities = softmax(dense_output)
print(f"   Output Shape: {probabilities.shape}")
prediction = np.argmax(probabilities)
confidence = probabilities[prediction]
print(f"The (untrained) network predicts: {prediction} with {confidence*100:.2f}% confidence.")
