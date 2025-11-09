
fn main() {
        let learning_rate: f64 = 0.005;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let dropout_rate: f64 = 0.3;
        let epsilon: f64 = 1e-8;
        let epochs: f64 = 30000.0;
        let lambda_l2: f64 = 0.001;
        let labels: f64 = 1.0;
        let alpha: f64 = 0.01;

       let mut input1: f64 = 1.0;
       let mut input2: f64 = 1.0;
       let mut input3: f64 = 1.0;
       let mut input4: f64 = 1.0;
       let mut weights1: f64 = 1.0;
       let mut weights2: f64 = 1.0;
       let mut weights3: f64 = 1.0;
       let mut weights4: f64 = 1.0;
       let mut bias: f64 = 0.0;
        
       if input1 < 0.0 {
          input1 = input1 * alpha;
        }
        if input2 < 0.0 {
          input2 = input2 * alpha;
        }
        if input3 < 0.0 {
           input3 = input3 * alpha;
        }
        if input4 < 0.0 {
            input4 = input4 * alpha;
        }
        
        let mut output: f64 = ((input1 * weights1) + (input2 * weights2) + (input3 * weights3) + (input4 * weights4)) + bias;
    println!("print the output:\n{}", output);

         output = output * (1.0 - dropout_rate);
    
    let exp_result: f64 = (-output).exp();
    let sigmoid_double: f64 = 1.0 / (1.0 + exp_result);
    println!("Sigmoid Activation:\n{}", sigmoid_double);
    
    let error: f64 = sigmoid_double - labels;
    println!("print the error:\n{}", error);
    
    let d_sigmoid: f64 = sigmoid_double * (1.0 - sigmoid_double);
    let gradient_at_output: f64 = error * d_sigmoid;
    println!("calculate:\n{}", gradient_at_output);
    
    let d_weights1: f64 = gradient_at_output * input1;
    let d_weights2: f64 = gradient_at_output * input2;
    let d_weights3: f64 = gradient_at_output * input3;
    let d_weights4: f64 = gradient_at_output * input4;
    let d_bias: f64 = gradient_at_output;
    
    weights1 = weights1 - (learning_rate * d_weights1);
    weights2 = weights2 - (learning_rate * d_weights2);
    weights3 = weights3 - (learning_rate * d_weights3);
    weights4 = weights4 - (learning_rate * d_weights4);
    bias = bias - (learning_rate * d_bias);
    
    println!("Neural Network in Rust\n");
}

