import numpy as np

# --- The Seed and the Functions of the Mind ---
# By setting the seed, we ensure that this "newborn" brain, while random,
# is the same newborn every time we run the code. This is the key to good science.
np.random.seed(42)

def sigmoid(x):
    # The neuron's decision: squashes any number into a 0-1 probability.
    # This is the "activation," the moment a neuron decides how strongly to fire.
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    # The soul of confidence: turns the final raw scores into a perfect
    # probability distribution for each input in the batch.
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# --- The Architecture of the Brain ---
# We define the shape of our thinking machine.
input_size = 2      # We will look at 2 pieces of evidence (e.g., score, study hours)
hidden_size = 4     # The first department will have 4 specialist detectives.
output_size = 2     # The final decision will be between 2 choices (e.g., Fail or Pass)

# --- Hiring the Detectives (Initializing the "Knobs") ---
# Here, we create the "newborn" brain. The weights are random whispers, and the biases are perfectly neutral.

# The clues for the first department of detectives.
hidden_weights = np.random.randn(input_size, hidden_size) * 0.01
hidden_biases = np.zeros(hidden_size)

# The clues for the final "Chief Detectives."
output_weights = np.random.randn(hidden_size, output_size) * 0.01
output_biases = np.zeros(output_size)


# --- The First Test Case: A Pre-written Batch ---

print("=" * 50)
print("RUNNING A PRE-DEFINED BATCH TEST")
print("=" * 50)
# 1. The Case Files (The Batch of Inputs)
inputs = np.array([[90, 17.5],  # Rohit's data
                   [70, 18.5]]) # Rohon's data
names = ["Rohit", "Rohon"]

# --- The Chain of Thought (The Forward Pass) ---

# 2. The Hidden Layer's Analysis
hidden_layer_inputs = np.dot(inputs, hidden_weights) + hidden_biases
hidden_layer_output = sigmoid(hidden_layer_inputs)
print(f"Hidden Layer Report (for both students):\n{np.round(hidden_layer_output, 4)}\n")

# 3. The Output Layer's Analysis
# The final detectives read the report from the hidden layer.
output_layer_inputs = np.dot(hidden_layer_output, output_weights) + output_biases
output_probabilities = softmax(output_layer_inputs)
print(f"Final Probabilities (Fail vs Pass):\n{np.round(output_probabilities, 4)}\n")

# 4. The Final Verdict
predictions = np.argmax(output_probabilities, axis=1)
for i in range(len(names)):
    verdict = "PASS ✅" if predictions[i] == 1 else "FAIL ❌"
    confidence = output_probabilities[i, predictions[i]] * 100
    print(f"Prediction for {names[i]}: {verdict} (Confidence: {confidence:.1f}%)")


# --- The Second Test Case: A Live Demonstration ---

print("\n" + "=" * 50)
print("LIVE PREDICTION FROM USER INPUT")
print("=" * 50)
try:
    user_inputs = np.array([
        [float(input("Enter Score for Student 1: ")), float(input("Enter Study Hours for Student 1: "))],
        [float(input("Enter Score for Student 2: ")), float(input("Enter Study Hours for Student 2: "))]
    ])
except ValueError:
    print("Invalid input. Please enter only numbers.")
    exit()

# The brain performs the exact same chain of thought on the new data.
user_hidden_inputs = np.dot(user_inputs, hidden_weights) + hidden_biases
user_hidden_output = sigmoid(user_hidden_inputs)

user_output_inputs = np.dot(user_hidden_output, output_weights) + output_biases
user_output_probabilities = softmax(user_output_inputs)

print("\n--- Final Verdicts for New Students ---")
user_predictions = np.argmax(user_output_probabilities, axis=1)
for i in range(len(names)):
    verdict = "PASS ✅" if user_predictions[i] == 1 else "FAIL ❌"
    confidence = user_output_probabilities[i, user_predictions[i]] * 100
    print(f"Prediction for Student {i+1}: {verdict} (Confidence: {confidence:.1f}%)")
import numpy as np
np.random.seed(42)
input_size = 2
hidden_size = 4
output_size = 2

score_weight1 = np.array([90])
study_hours_bias1 = np.array([17.5])
output1 = score_weight1 + study_hours_bias1

score_weight2 = np.array([70])
study_hours_bias2 = np.array([18.5])
output2 = score_weight2 + study_hours_bias2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

inputs = np.array([[90, 20.5],
                   [80, 18.5]])
names = ["Student1", "Student2"]
hidden_weights = np.random.randn(input_size, hidden_size) * 0.01
hidden_biases = np.zeros(hidden_size)

output_weights = np.random.randn(hidden_size, output_size) * 0.01
output_biases = np.zeros(output_size)

# Forward pass - Hidden layer
hidden_layer_inputs = np.dot(inputs, hidden_weights) + hidden_biases
hidden_layer_output = sigmoid(hidden_layer_inputs)
print(f"Hidden Layer Output:\n{np.round(hidden_layer_output, 2)}\n")

# Forward pass - Output layer
output_layers_inputs = np.dot(hidden_layer_output, output_weights) + output_biases
output_probabilities = softmax(output_layers_inputs)
print(f"Output Probabilities (softmax):\n{np.round(output_probabilities, 2)}\n")


predictions = np.argmax(output_probabilities, axis=1)

for i in range(len(names)):
    verdict = "PASS" if predictions[i] == 1 else "FAIL"
    confidence = output_probabilities[i, predictions[i]] * 100
    print(f"Prediction for {names[i]}: {verdict}")
    print(f"(Confidence: {confidence:.1f}%)\n")
# data prediction

user_inputs = np.array([
        [float(input("Enter Score for Student1: ")), float(input("Enter Study Hours for Student1: "))],
        [float(input("Enter Score for Student2: ")), float(input("Enter Study Hours for Student2: "))]
    ])
user_hidden_inputs = np.dot(user_inputs, hidden_weights) + hidden_biases
user_hidden_output = sigmoid(user_hidden_inputs)
print(f"Hidden Layer Output:\n{np.round(user_hidden_output, 2)}\n")
output_layers_inputs = np.dot(user_hidden_output, output_weights) + output_biases
user_output_probabilities = softmax(output_layers_inputs)
print(f"Output Probabilities (softmax):\n{np.round(user_output_probabilities, 2)}\n")
user_predictions = np.argmax(user_output_probabilities, axis=1)
for i in range(len(names)):
    verdict = "PASS" if user_predictions[i] == 1 else "FAIL"
    confidence = user_output_probabilities[i, user_predictions[i]] * 100
    print(f"Prediction for {names[i]}: {verdict}")
    print(f"(Confidence: {confidence:.1f}%)\n")
