let learning_rate = 0.005;
let labels = 1.0;
let alpha = 0.01;
let epochs = 100;

const input1_orig = 1.0;
const input2_orig = -2.0;
const input3_orig = 3.0;
const input4_orig = -4.0;

let weights1 = -5.9;
let weights2 = 6.8;
let weights3 = -7.8;
let weights4 = 8.0;
let bias = 0.0;

console.log("Starting Training...\n");

for (let epoch = 0; epoch < epochs; epoch++) {
    let input1 = (input1_orig > 0) ? input1_orig : input1_orig * alpha;
    let input2 = (input2_orig > 0) ? input2_orig : input2_orig * alpha;
    let input3 = (input3_orig > 0) ? input3_orig : input3_orig * alpha;
    let input4 = (input4_orig > 0) ? input4_orig : input4_orig * alpha;

    let output = (input1 * weights1) + (input2 * weights2) + 
                 (input3 * weights3) + (input4 * weights4) + bias;

    let sigmoid = 1.0 / (1.0 + Math.exp(-output));
    let error = sigmoid - labels;
    
    let d_sigmoid = sigmoid * (1 - sigmoid);
    let gradient = error * d_sigmoid;

    weights1 -= learning_rate * gradient * input1;
    weights2 -= learning_rate * gradient * input2;
    weights3 -= learning_rate * gradient * input3;
    weights4 -= learning_rate * gradient * input4;
    bias -= learning_rate * gradient;

    if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}: Output = ${sigmoid.toFixed(4)}, Error = ${error.toFixed(4)}`);
    }
}

console.log("\n=== Training Complete ===");
console.log(`Final weights: ${weights1.toFixed(4)}, ${weights2.toFixed(4)}, ${weights3.toFixed(4)}, ${weights4.toFixed(4)}`);
console.log(`Final bias: ${bias.toFixed(4)}`);
