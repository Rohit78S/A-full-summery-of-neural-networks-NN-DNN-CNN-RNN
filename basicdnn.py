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
    dataset.append([score, hours, 0, 1])
num_fail_med = int(NUM_FAIL * 0.57); num_fail_low = NUM_FAIL - num_fail_med
for i in range(num_fail_med): 
    score = np.random.uniform(40, 60); hours = np.random.uniform(1, 20)
    dataset.append([score, hours, 1, 0])
for i in range(num_fail_low): 
    score = np.random.uniform(20, 40); hours = np.random.uniform(0.5, 10)
    dataset.append([score, hours, 1, 0])

np.random.shuffle(dataset)
df = pd.DataFrame(dataset, columns=['Score', 'Study_Hours', 'Label_Fail', 'Label_Pass'])
df.to_csv(FILENAME, index=False)
print(f"Dataset saved to '{FILENAME}' ({len(df)} samples)")
print(f"PASS samples: {(df['Label_Pass'] == 1).sum()}")
print(f"FAIL samples: {(df['Label_Fail'] == 1).sum()}\n")


print("Reading data from CSV...")
df = pd.read_csv(FILENAME)
inputs = df[['Score', 'Study_Hours']].values
labels = df[['Label_Fail', 'Label_Pass']].values
mean = inputs.mean(axis=0); std = inputs.std(axis=0)
inputs_scaled = (inputs - mean) / std 
print(f"Loaded {len(inputs)} samples from CSV")
print(f"Input shape (scaled): {inputs_scaled.shape}")
print(f"Labels shape: {labels.shape}\n")

input_size = 2
hidden_size1 = 16
hidden_size2 = 16
output_size = 2

learning_rate = 0.005
beta1 = 0.9
beta2 = 0.999
dropout_rate = 0.3
epsilon = 1e-8
epochs = 30000
lambda_l2 = 0.001

hidden_weights1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
hidden_biases1 = np.zeros(hidden_size1)
hidden_weights2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
hidden_biases2 = np.zeros(hidden_size2)
output_weights = np.random.randn(hidden_size2, output_size) * np.sqrt(1.0 / hidden_size2)
output_biases = np.zeros(output_size)

m_h_w1 = np.zeros_like(hidden_weights1)
m_h_b1 = np.zeros_like(hidden_biases1)
m_h_w2 = np.zeros_like(hidden_weights2)
m_h_b2 = np.zeros_like(hidden_biases2)
m_o_w = np.zeros_like(output_weights)
m_o_b = np.zeros_like(output_biases)

v_h_w1 = np.zeros_like(hidden_weights1)
v_h_b1 = np.zeros_like(hidden_biases1)
v_h_w2 = np.zeros_like(hidden_weights2)
v_h_b2 = np.zeros_like(hidden_biases2)
v_o_w = np.zeros_like(output_weights)
v_o_b = np.zeros_like(output_biases)
t = 0

def leaky_relu(x, alpha=0.01):
    return np.where( x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def adam_update(param, grad, m, v, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v
def forward_pass(data, training=True):
    hidden_inputs1 = np.dot(data, hidden_weights1) + hidden_biases1
    hidden_outputs1_activated = leaky_relu(hidden_inputs1)
    dropout_mask1 = None
    if training:
        dropout_mask1 = np.random.binomial(1, 1 - dropout_rate, size=hidden_outputs1_activated.shape) / (1 - dropout_rate)
        hidden_outputs1 = hidden_outputs1_activated * dropout_mask1
    else:
        hidden_outputs1 = hidden_outputs1_activated
    hidden_inputs2 = np.dot(hidden_outputs1, hidden_weights2) + hidden_biases2
    hidden_outputs2_activated = leaky_relu(hidden_inputs2)
    dropout_mask2 = None
    if training:
        dropout_mask2 = np.random.binomial(1, 1 - dropout_rate, size=hidden_outputs2_activated.shape) / (1 - dropout_rate)
        hidden_outputs2 = hidden_outputs2_activated * dropout_mask2
    else:
        hidden_outputs2 = hidden_outputs2_activated
    output_inputs = np.dot(hidden_outputs2, output_weights) + output_biases
    output_probs = softmax(output_inputs)
    return (output_probs,
            hidden_inputs1, hidden_outputs1_activated, dropout_mask1,
            hidden_inputs2, hidden_outputs2_activated, dropout_mask2)

def backward_pass_adam_l2_dropout(data, labels, output_probs,
            hidden_inputs1, hidden_outputs1_activated, dropout_mask1,
            hidden_inputs2, hidden_outputs2_activated, dropout_mask2):
    global  hidden_weights1, hidden_biases1, hidden_weights2, hidden_biases2, output_weights, output_biases
    global m_h_w1, m_h_b1, m_h_w2, m_h_b2, m_o_w, m_o_b
    global v_h_w1, v_h_b1, v_h_w2, v_h_b2, v_o_w, v_o_b
    global t
    num_samples = data.shape[0]
    t += 1
    output_error = output_probs - labels
    output_delta = output_error
    d_output_weights = np.dot((hidden_outputs2_activated * dropout_mask2).T, output_delta) / num_samples
    d_output_biases = np.sum(output_delta, axis=0) / num_samples
    d_output_weights += (lambda_l2 / num_samples) * output_weights

    hidden_error2 = np.dot(output_delta, output_weights.T)
    hidden_error2 *= dropout_mask2
    hidden_delta2 = hidden_error2 * leaky_relu_derivative(hidden_inputs2)
    d_hidden_weights2 = np.dot((hidden_outputs1_activated * dropout_mask1).T, hidden_delta2) / num_samples
    d_hidden_biases2 = np.sum(hidden_delta2, axis=0) / num_samples
    d_hidden_weights2 += (lambda_l2 / num_samples) * hidden_weights2

    hidden_error1 = np.dot(hidden_delta2, hidden_weights2.T)
    hidden_error1 *= dropout_mask1
    hidden_delta1 = hidden_error1 * leaky_relu_derivative(hidden_inputs1)
    d_hidden_weights1 = np.dot(data.T, hidden_delta1) / num_samples
    d_hidden_biases1 = np.sum(hidden_delta1, axis=0) / num_samples
    d_hidden_weights1 += (lambda_l2 / num_samples) * hidden_weights1

    output_weights, m_o_w, v_o_w = adam_update(output_weights, d_output_weights, m_o_w, v_o_w, t)
    output_biases, m_o_b, v_o_b = adam_update(output_biases, d_output_biases, m_o_b, v_o_b, t)
    hidden_weights2, m_h_w2, v_h_w2 = adam_update(hidden_weights2, d_hidden_weights2, m_h_w2, v_h_w2, t)
    hidden_biases2, m_h_b2, v_h_b2 = adam_update(hidden_biases2, d_hidden_biases2, m_h_b2, v_h_b2, t)
    hidden_weights1, m_h_w1, v_h_w1 = adam_update(hidden_weights1, d_hidden_weights1, m_h_w1, v_h_w1, t)
    hidden_biases1, m_h_b1, v_h_b1 = adam_update(hidden_biases1, d_hidden_biases1, m_h_b1, v_h_b1, t)

def calculate_loss(output_probs, labels):
    num_samples = labels.shape[0]
    epsilon = 1e-8
    correct_logprobs = -np.sum(labels * np.log(output_probs + epsilon), axis=1)
    data_loss = np.mean(correct_logprobs)
    l2_loss = (lambda_l2 / (2 * num_samples)) * (np.sum(np.square(hidden_weights1)) +
                                                  np.sum(np.square(hidden_weights2)) +
                                                  np.sum(np.square(output_weights)))
    total_loss = data_loss + l2_loss
    return total_loss


print("Training with Adam, L2, and Dropout...\n")
for epoch in range(epochs):
    (output_probs,
     hidden_inputs1, hidden_outputs1_act, mask1,
     hidden_inputs2, hidden_outputs2_act, mask2) = forward_pass(inputs_scaled, training=True)

    backward_pass_adam_l2_dropout(inputs_scaled, labels, output_probs,
                                  hidden_inputs1, hidden_outputs1_act, mask1,
                                  hidden_inputs2, hidden_outputs2_act, mask2)

    if (epoch + 1) % (epochs // 10) == 0: # Print 10 times
        # Run forward pass WITHOUT dropout for accurate loss/accuracy
        output_probs_no_dropout, _, _, _, _, _, _ = forward_pass(inputs_scaled, training=False)
        loss = calculate_loss(output_probs_no_dropout, labels) 
        predictions = np.argmax(output_probs_no_dropout, axis=1)
        accuracy = np.mean(predictions == np.argmax(labels, axis=1)) * 100
        print(f"Epoch: {epoch + 1:6,}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")

print("\nTraining Complete!\n")


print("--- Predictions on first 5 training samples (Dropout OFF) ---")
output_probs_train, _, _, _, _, _, _ = forward_pass(inputs_scaled[:5], training=False)
predictions_train = np.argmax(output_probs_train, axis=1)
original_inputs_train = inputs[:5]
for i in range(len(original_inputs_train)):
    verdict = "PASS" if predictions_train[i] == 1 else "FAIL"
    confidence = output_probs_train[i, predictions_train[i]] * 100
    actual_verdict = 'PASS' if labels[i,1] == 1 else 'FAIL'
    match = " " if verdict == actual_verdict else "✗"
    print(f"{match} Sample {i+1} (Input: {np.round(original_inputs_train[i],1)}): Predicted {verdict} (Conf: {confidence:.1f}%) -- Actual: {actual_verdict}")

# On New User Data
print("\n--- Predictions on New User Data (Dropout OFF) ---")
user_inputs_list = []
names = ["Student A", "Student B"] 
for name in names:
     score = float(input(f"Enter Score for {name}: "))
     hours = float(input(f"Enter Study Hours for {name}: "))
     user_inputs_list.append([score, hours])
user_inputs_raw = np.array(user_inputs_list)
user_inputs_scaled = (user_inputs_raw - mean) / std 

user_output_probs, _, _, _, _, _, _ = forward_pass(user_inputs_scaled, training=False) 
user_predictions = np.argmax(user_output_probs, axis=1)

for i in range(len(names)):
    verdict = "PASS" if user_predictions[i] == 1 else "FAIL"
    confidence = user_output_probs[i, user_predictions[i]] * 100
    print(f"\nPrediction for {names[i]}: {verdict}")
    print(f"  (Raw Input: Score={user_inputs_raw[i,0]:.1f}, Hours={user_inputs_raw[i,1]:.1f})")
    print(f"  (Confidence: {confidence:.1f}%)")
    print(f"  (Probabilities: [Fail: {user_output_probs[i, 0]*100:.1f}%, Pass: {user_output_probs[i, 1]*100:.1f}%])")
