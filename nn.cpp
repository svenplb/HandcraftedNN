#include <vector>
#include <iostream>
#include <algorithm>

class Layer
{
public:
    // on the middle layer of a (2,3,2) neural net nodesIn will be 2, nodesOut will be 3
    int numNodesIn, numNodesOut;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(int numNodesIn, int numNodesOut)
    {
        this->numNodesIn = numNodesIn;
        this->numNodesOut = numNodesOut;

        // initialize weights and biases with 0
        weights = std::vector<std::vector<double>>(numNodesIn, std::vector<double>(numNodesOut, 0.0));
        biases = std::vector<double>(numNodesOut, 0.0);
    }

    std::vector<double> CalculateOutputs(const std::vector<double> &inputs)
    {
        // create vector of output values, initialize with 0
        std::vector<double> activations(numNodesOut, 0.0);

        // calculate weighed sum of each output node
        // outputNode = (input1*weight1) + (input1*weight2) + (input2*weight3) + (input2*weight_2_1)
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                // calculate weighted sum of inputNode and add it to the weightedSum
                weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
            }
            // add WeightedSum of output node to the weightedInputs
            activations[nodeOut] = ActivationFunction(weightedInput);
        }

        return activations;
    }
    double ActivationFunction(double weightedInput)
    {
        return (weightedInput > 0) ? weightedInput : 0;
    }
};

class NeuralNetwork
{
public:
    // intialize vector of the layers class
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int> &layerSizes)
    {
        // create layers based on defined layer size and add it to the layers array
        for (size_t i = 0; i < layerSizes.size() - 1; i++)
        {
            layers.emplace_back(Layer(layerSizes[i], layerSizes[i + 1]));
        }
    }

    // take input, feed through neural net & calculate outputs
    std::vector<double> feedForward(std::vector<double> inputs)
    {
        for (auto &layer : layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    int Classify(std::vector<double> inputs)
    {
        std::vector<double> outputs = feedForward(inputs);
        return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    }
};

int main(void)
{
    std::vector<int> layerSizes = {2, 3, 2};
    NeuralNetwork network(layerSizes);
}
