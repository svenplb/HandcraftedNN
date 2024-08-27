#include <vector>
#include <iostream>

class Layer {
public:
    int numNodesIn, numNodesOut;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    // Constructor to create the layer
    Layer(int numNodesIn, int numNodesOut) {
        this->numNodesIn = numNodesIn;
        this->numNodesOut = numNodesOut;

        // Initialize weights and biases
        weights = std::vector<std::vector<double>>(numNodesIn, std::vector<double>(numNodesOut, 0.0));
        biases = std::vector<double>(numNodesOut, 0.0);
    }

    // Calculate the output of the layer
    std::vector<double> CalculateOutputs(const std::vector<double>& inputs) {
        std::vector<double> weightedInputs(numNodesOut, 0.0);

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        return weightedInputs;
    }
};

int main(void) {

}
