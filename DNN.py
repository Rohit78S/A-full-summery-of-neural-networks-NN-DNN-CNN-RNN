import numpy as np
import pandas as pd

FILENAME = 'student_data_1000.csv'
NUM_SAMPLES = 1000
PASS_RATIO = 0.65
NUM_PASS = int(NUM_SAMPLES * PASS_RATIO)
NUM_FAIL = NUM_SAMPLES - NUM_PASS

print(f"Generating {NUM_SAMPLES} sample dataset...")
np.random.seed(42)
dataset = []

for i in range(NUM_PASS):
    score = np.random.uniform(70, 100); hours = np.random.uniform(1, 20)
    dataset.append([round(score,2), round(hours,2), 0, 1])
num_fail_med = int(NUM_FAIL * 0.57); num_fail_low = NUM_FAIL - num_fail_med
for i in range(num_fail_med):
    score = np.random.uniform(40, 60); hours = np.random.uniform(1, 20)
    dataset.append([round(score,2), round(hours,2), 1, 0])
for i in range(num_fail_low):
    score = np.random.uniform(20, 40); hours = np.random.uniform(0.5, 10)
    dataset.append([round(score,2), round(hours,2), 1, 0])
np.random.shuffle(dataset)
df = pd.DataFrame(dataset, columns=['Score', 'Study_Hours', 'Label_Fail', 'Label_Pass'])
df.to_csv(FILENAME, index=False)
print(f"Dataset saved to '{FILENAME}' ({len(df)} samples)")
print(f"PASS samples: {(df['Label_Pass'] == 1).sum()}")
print(f"FAIL samples: {(df['Label_Fail'] == 1).sum()}\n")


print("Reading data from CSV...")
df = pd.read_csv(FILENAME)
inputs = df[['Score', 'Study_Hours']].values
labels_one_hot = df[['Label_Fail', 'Label_Pass']].values
labels_indices = np.argmax(labels_one_hot, axis=1) 

mean = inputs.mean(axis=0); std = inputs.std(axis=0)
inputs_scaled = (inputs - mean) / std 
print(f"Loaded {len(inputs)} samples from CSV")
print(f"Input shape (scaled): {inputs_scaled.shape}")
print(f"Labels shape (indices): {labels_indices.shape}\n")


input_size = 2
hidden_size1 = 16
hidden_size2 = 16
output_size = 2

learning_rate = 0.005
epochs = 30000
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
lambda_l2_weights = 0.001
lambda_l2_biases = 0 
dropout_rate = 0.3



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_lambda_l2=0, bias_lambda_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs) 
        self.biases = np.zeros((1, n_neurons))
        self.weight_lambda_l2 = weight_lambda_l2
        self.bias_lambda_l2 = bias_lambda_l2
        self.inputs = None; self.output = None
        self.dweights = None; self.dbiases = None; self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

       
        if self.weight_lambda_l2 > 0:    
            self.dweights += 2 * self.weight_lambda_l2 * self.weights
        if self.bias_lambda_l2 > 0:
            self.dbiases += 2 * self.bias_lambda_l2 * self.biases

       
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.inputs = None; self.output = None; self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, dvalues):
        d_leaky_relu = np.where(self.inputs > 0, 1, self.alpha)
        self.dinputs = dvalues * d_leaky_relu

class Activation_Softmax:
    def __init__(self):
        self.inputs = None; self.output = None; self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = rate 
        self.keep_probability = 1 - rate
        self.mask = None
        self.inputs = None; self.output = None; self.dinputs = None
        self._training_mode = True

    def forward(self, inputs):
        self.inputs = inputs
        if not self._training_mode:
            self.output = inputs.copy()
            return

        self.mask = np.random.binomial(1, self.keep_probability, size=inputs.shape) / self.keep_probability
        self.output = inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask

    def set_prediction_mode(self): self._training_mode = False
    def set_training_mode(self): self._training_mode = True

class Loss:
    def calculate(self, output, y, layers): 
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        reg_loss = self.regularization_loss(layers)
        return data_loss + reg_loss

    def regularization_loss(self, layers):
        reg_loss = 0
        for layer in layers:
            if isinstance(layer, Layer_Dense): 
                if layer.weight_lambda_l2 > 0:
                    reg_loss += layer.weight_lambda_l2 * np.sum(layer.weights * layer.weights)
                if layer.bias_lambda_l2 > 0:
                    reg_loss += layer.bias_lambda_l2 * np.sum(layer.biases * layer.biases)
        return 0.5 * reg_loss 

class Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError("Invalid shape for y_true")

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """Backward pass for loss."""
        samples = len(dvalues)
        num_outputs = len(dvalues[0])

        
        if len(y_true.shape) == 1:
            y_true_one_hot = np.eye(num_outputs)[y_true]
        elif len(y_true.shape) == 2:
             y_true_one_hot = y_true 
        else:
             raise ValueError("Invalid shape for y_true")

        
        self.dinputs = -y_true_one_hot / dvalues
        
        self.dinputs = self.dinputs / samples

-
class Activation_Softmax_Loss_CategoricalCrossentropy():
    """Combines Softmax activation and CrossEntropy loss for stable backward pass."""
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        self.output = None; self.dinputs = None

    def forward(self, inputs, y_true):
        """Forward pass: activate, calculate loss."""
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true) 

    def backward(self, dvalues, y_true):
        """Backward pass using simplified gradient for Softmax+CrossEntropy."""
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true_indices = np.argmax(y_true, axis=1)
        elif len(y_true.shape) == 1:
            y_true_indices = y_true 
        else:
             raise ValueError("Invalid shape for y_true")

       
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true_indices] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentums = {} 
        self.caches = {}    

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Update parameters for a dense layer."""
        layer_id = id(layer) 

        
        if layer_id not in self.momentums:
            self.momentums[layer_id] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}
            self.caches[layer_id] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}

       
        self.momentums[layer_id]['weights'] = self.beta_1 * self.momentums[layer_id]['weights'] + (1 - self.beta_1) * layer.dweights
        self.momentums[layer_id]['biases'] = self.beta_1 * self.momentums[layer_id]['biases'] + (1 - self.beta_1) * layer.dbiases

        
        t = self.iterations + 1
        momentum_corrected_weights = self.momentums[layer_id]['weights'] / (1 - self.beta_1 ** t)
        momentum_corrected_biases = self.momentums[layer_id]['biases'] / (1 - self.beta_1 ** t)

        
        self.caches[layer_id]['weights'] = self.beta_2 * self.caches[layer_id]['weights'] + (1 - self.beta_2) * layer.dweights**2
        self.caches[layer_id]['biases'] = self.beta_2 * self.caches[layer_id]['biases'] + (1 - self.beta_2) * layer.dbiases**2

       
        cache_corrected_weights = self.caches[layer_id]['weights'] / (1 - self.beta_2 ** t)
        cache_corrected_biases = self.caches[layer_id]['biases'] / (1 - self.beta_2 ** t)

        
        layer.weights -= self.current_learning_rate * momentum_corrected_weights / (np.sqrt(cache_corrected_weights) + self.epsilon)
        layer.biases -= self.current_learning_rate * momentum_corrected_biases / (np.sqrt(cache_corrected_biases) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

dense1 = Layer_Dense(input_size, hidden_size1, weight_lambda_l2=lambda_l2_weights)
activation1 = Activation_LeakyReLU(alpha=0.01)
dropout1 = Layer_Dropout(dropout_rate)
dense2 = Layer_Dense(hidden_size1, hidden_size2, weight_lambda_l2=lambda_l2_weights)
activation2 = Activation_LeakyReLU(alpha=0.01)
dropout2 = Layer_Dropout(dropout_rate)
dense3_output = Layer_Dense(hidden_size2, output_size, weight_lambda_l2=lambda_l2_weights)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=learning_rate / epochs)


trainable_layers = [dense1, dense2, dense3_output]


print("Training with Class-based Adam, L2, and Dropout...\n")
for epoch in range(epochs + 1):
    dropout1.set_training_mode(); dropout2.set_training_mode()


    dense1.forward(inputs_scaled)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    dense3_output.forward(dropout2.output)
    sample_losses = loss_activation.forward(dense3_output.output, labels_indices) 

    data_loss = np.mean(sample_losses)
    regularization_loss = loss_activation.loss.regularization_loss(trainable_layers)
    total_loss = data_loss + regularization_loss

    
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == labels_indices) * 100

    
    loss_activation.backward(loss_activation.output, labels_indices) 
    dense3_output.backward(loss_activation.dinputs)
    dropout2.backward(dense3_output.dinputs)
    activation2.backward(dropout2.dinputs)
    dense2.backward(activation2.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    
    optimizer.pre_update_params()
    for layer in trainable_layers:
        optimizer.update_params(layer)
    optimizer.post_update_params()

    
    if epoch % (epochs // 20) == 0 or epoch == epochs: 
        print(f"Epoch: {epoch:6,}/{epochs}, "
              f"Acc: {accuracy:.2f}%, "
              f"Loss: {total_loss:.6f} (Data: {data_loss:.6f}, Reg: {regularization_loss:.6f}), "
              f"LR: {optimizer.current_learning_rate:.6f}")

print("\nTraining Complete!\n")

dropout1.set_prediction_mode(); dropout2.set_prediction_mode()


print("--- Predictions on first 5 training samples (Dropout OFF) ---")
dense1.forward(inputs_scaled[:5])
activation1.forward(dense1.output)
dropout1.forward(activation1.output)
dense2.forward(dropout1.output)
activation2.forward(dense2.output)
dropout2.forward(activation2.output)
dense3_output.forward(dropout2.output)
pred_softmax = Activation_Softmax()
pred_softmax.forward(dense3_output.output)
output_probs_train = pred_softmax.output

predictions_train = np.argmax(output_probs_train, axis=1)
original_inputs_train = inputs[:5]
for i in range(len(original_inputs_train)):
    verdict = "PASS" if predictions_train[i] == 1 else "FAIL"
    confidence = output_probs_train[i, predictions_train[i]] * 100
    actual_verdict = 'PASS' if labels_indices[i] == 1 else 'FAIL' 
    match = " " if verdict == actual_verdict else "✗"
    print(f"{match} Sample {i+1} (Input: {np.round(original_inputs_train[i],1)}): Predicted {verdict} (Conf: {confidence:.1f}%) -- Actual: {actual_verdict}")


print("\n--- Predictions on New User Data (Dropout OFF) ---")
user_inputs_list = []
names = ["Student A", "Student B"]
for name in names:
     score = float(input(f"Enter Score for {name}: "))
     hours = float(input(f"Enter Study Hours for {name}: "))
     user_inputs_list.append([score, hours])
user_inputs_raw = np.array(user_inputs_list)
user_inputs_scaled = (user_inputs_raw - mean) / std 

dense1.forward(user_inputs_scaled)
activation1.forward(dense1.output)
dropout1.forward(activation1.output)
dense2.forward(dropout1.output)
activation2.forward(dense2.output)
dropout2.forward(activation2.output)
dense3_output.forward(dropout2.output)
pred_softmax.forward(dense3_output.output)
user_output_probs = pred_softmax.output

user_predictions = np.argmax(user_output_probs, axis=1)

for i in range(len(names)):
    verdict = "PASS" if user_predictions[i] == 1 else "FAIL"
    confidence = user_output_probs[i, user_predictions[i]] * 100
    print(f"\nPrediction for {names[i]}: {verdict}")
    print(f"  (Raw Input: Score={user_inputs_raw[i,0]:.1f}, Hours={user_inputs_raw[i,1]:.1f})")
    print(f"  (Confidence: {confidence:.1f}%)")
    print(f"  (Probabilities: [Fail: {user_output_probs[i, 0]*100:.1f}%, Pass: {user_output_probs[i, 1]*100:.1f}%])")
