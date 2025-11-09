//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {

        float learning_rate = 0.005f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float dropout_rate = 0.3f;
        float epsilon = 1e-8f;
        float epochs = 30000f;
        float lambda_l2 = 0.001f;
        float labels = 1.0f;
        float alpha = 0.01f;

        float input1 = 1;
        float input2 = 1;
        float input3 = 1;
        float input4 = 1;

        float weights1 = 1;
        float weights2 = 1;
        float weights3 = 1;
        float weights4 = 1;

        float bias = 0;


        if (input1 < 0) {
            input1 = (input1 * alpha);
        }
        if (input2 < 0) {
            input2 = (input2 * alpha);
        }
        if (input3 < 0) {
            input3 = (input3 * alpha);
        }
        if (input4 < 0) {
            input4 = (input4 * alpha);
        }

        float output = ((input1 * weights1) + (input2 * weights2) + (input3 * weights3) + (input4 * weights4)) + bias;
        System.out.println("print the output:\n" + output);

        output = output * (1 - dropout_rate);
        double expResult = Math.exp(-output);
        double sigmoidDouble = 1.0 / (1.0 + expResult);
        System.out.println("Sigmoid Activation:\n" + sigmoidDouble);

        double error = (sigmoidDouble - labels);
        System.out.println("print the error:\n" + error);

        double d_sigmoid = (sigmoidDouble * (1 - sigmoidDouble));
        double gradient_at_output = error * d_sigmoid;
        System.out.println("calculate:\n" + gradient_at_output);

        double d_weights1 = (gradient_at_output * input1);
        double d_weights2 = (gradient_at_output * input2);
        double d_weights3 = (gradient_at_output * input3);
        double d_weights4 = (gradient_at_output * input4);

        double d_bias = gradient_at_output;

        weights1 = (float) (weights1 - (learning_rate * d_weights1));
        weights2 = (float) (weights2 - (learning_rate * d_weights2));
        weights3 = (float) (weights3 - (learning_rate * d_weights3));
        weights4 = (float) (weights4 - (learning_rate * d_weights4));

        bias = (float) (bias - (learning_rate * d_bias));

        System.out.println("Neuran Network in java\n");


    }

}
