#include <vector>
#include <iostream>
#include <algorithm>

class Layer {
public:
    int numNodesIn, numNodesOut;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(int numNodesIn, int numNodesOut) {
        this->numNodesIn = numNodesIn;
        this->numNodesOut = numNodesOut;

        weights = std::vector<std::vector<double>>(numNodesIn, std::vector<double>(numNodesOut, 0.0));
        biases = std::vector<double>(numNodesOut, 0.0);
    }

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

class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& layerSizes) {
        for (size_t i = 0; i < layerSizes.size() - 1; i++) {
            layers.emplace_back(Layer(layerSizes[i], layerSizes[i + 1]));
        }
    }

    std::vector<double> CalculateOutputs(std::vector<double> inputs) {
        for (auto& layer : layers) {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    int Classify(std::vector<double> inputs) {
        std::vector<double> outputs = CalculateOutputs(inputs);
        return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    }
};


int main(void) {
	std::vector<int> layerSizes = {2, 3, 2};
	NeuralNetwork network(layerSizes);
}

