import numpy as np
np.random.seed(42)
#input values user define
inputs = np.array([
    float(input("Value 1: ")),
    float(input("Value 2: ")),
    float(input("Value 3: ")),
    float(input("Value 4: "))
])
#funcation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    exp_scores = np.exp(x - np.max(x))
    return exp_scores / exp_scores.sum()
# Hidden Layer 1
hidden_weights = np.random.uniform(0, 400, size=(4, 4))
hidden_bias = np.array([1, 2, 3, 4])
hidden_activation = np.dot(inputs, hidden_weights) + hidden_bias
hidden_output = sigmoid(hidden_activation)
print(f"Output: {np.round(hidden_output, 4)}")
# Hidden Layer 2
hidden_weights2 = np.random.uniform(0, 400, size=(4, 4))
hidden_bias2 = np.array([5, 6, 7, 8])
hidden_activation2 = np.dot(hidden_output, hidden_weights2) + hidden_bias2
hidden_output2 = sigmoid(hidden_activation2)
print(f"Output: {np.round(hidden_output2, 4)}")
# Hidden Layer 3
hidden_weights3 = np.random.uniform(0, 400, size=(4, 4))
hidden_bias3 = np.array([9, 10, 11, 12])
hidden_activation3 = np.dot(hidden_output2, hidden_weights3) + hidden_bias3
hidden_output3 = sigmoid(hidden_activation3)
print(f"Output: {np.round(hidden_output3, 4)}")
hidden_weights4 = np.random.uniform(0, 400, size=(4, 4))
hidden_bias4 = np.array([9, 10, 11, 12])
hidden_activation4 = np.dot(hidden_output3, hidden_weights4) + hidden_bias4
hidden_output4 = sigmoid(hidden_activation4)
print(f"Output: {np.round(hidden_output4, 4)}")

# Output Layer - Cat Neuron
cat_weights = np.random.uniform(0, 400, size=4)
cat_bias = 1
cat_activation = np.dot(cat_weights, hidden_output4) + cat_bias
print(f"\nRandomly generated Cat weights: {np.round(cat_weights, 2)}")
print(f"Cat Specialist's Confidence Score: {cat_activation:.2f}")

# Output Layer - Dog Neuron
dog_weights = np.random.uniform(0, 400, size=4)
dog_bias = 1
dog_activation = np.dot(dog_weights, hidden_output4) + dog_bias
print(f"\n Randomly generated Dog weights: {np.round(dog_weights, 2)}")
print(f"Dog Specialist's Confidence Score: {dog_activation:.2f}")
raw_scores = np.array([cat_activation, dog_activation])
confidences = softmax(raw_scores)
cat_confidence = confidences[0] * 100
dog_confidence = confidences[1] * 100

if cat_confidence > dog_confidence:
    print("\nPrediction: This is a CAT")
    print(f"(Confidence: {cat_confidence:.1f}%)")
else:
    print("\nPrediction: This is a DOG")
    print(f"(Confidence: {dog_confidence:.1f}%)")
