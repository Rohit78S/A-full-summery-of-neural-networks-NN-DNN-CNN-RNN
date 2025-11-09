#include <stdio.h>
#include <math.h>

int main() {
    float learning_rate = 0.005;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float dropout_rate = 0.3;
    float epsilon = 1e-8;
    float epochs = 30000;
    float lambda_l2 = 0.001;
    float labels = 1.0;

    float input1 = 1.0;
    float input2 = -2.0;
    float input3 = 3.0;
    float input4 = -4.0;
    float input_sum = input1 + input2 + input3 + input4;

    float weights1 = -5.9;
    float weights2 = 6.8;
    float weights3 = -7.8;
    float weights4 = 8.0;
    float weights_sum = weights1 + weights2 + weights3 + weights4;
    float bias = 0.0;

    float alpha = 0.01;

    if (input1 < 0) {
        input1 = input1 * alpha;
    }
    if (input2 < 0) {
        input2 = input2 * alpha;
    }
    if (input3 < 0) {
        input3 = input3 * alpha;
    }
    if (input4 < 0) {
        input4 = input4 * alpha;
    }




    float output = ((input1 * weights1)  + (input2 * weights2) + (input3 * weights3)+ (input4 * weights4)) + bias;
    printf("Output = %f\n", output);
    output = output * (1 - dropout_rate);
    printf("Output (Scaled) = %f\n", output);



    printf("Output (Scaled) = %f\n", output);

    float sigmoid = 1.0 / (1.0 + exp(-output));
    printf("Sigmoid Activation = %f\n", sigmoid);

    float error = sigmoid - labels;
    printf("Error = %f\n", error);
    float d_sigmoid = sigmoid * (1.0 - sigmoid);
    float gradient_at_output = error * d_sigmoid;

    float d_weight1 = gradient_at_output * input1;
    float d_weight2 = gradient_at_output * input2;
    float d_weight3 = gradient_at_output * input3;
    float d_weight4 = gradient_at_output * input4;

    float d_bias = gradient_at_output; // (Multiplied by 1)

    weights1 = weights1 - (learning_rate * d_weight1);
    weights2 = weights2 - (learning_rate * d_weight2);
    weights3 = weights3 - (learning_rate * d_weight3);
    weights4 = weights4 - (learning_rate * d_weight4);
    bias     = bias - (learning_rate * d_bias);



    return 0;

}
