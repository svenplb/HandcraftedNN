// TODO: Better Comments, Clean Up, Use actual data, training.cpp file, refactor

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

class Layer
{
public:
    // on the middle layer of a (2,3,2) neural net nodesIn will be 2, nodesOut will be 3
    int numNodesIn, numNodesOut;

    // cost gradients weights and biases
    std::vector<std::vector<double>> costGradientWeights;
    std::vector<double> costGradientBias;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(int numNodesIn, int numNodesOut)
    {
        this->numNodesIn = numNodesIn;
        this->numNodesOut = numNodesOut;

        costGradientWeights = std::vector<std::vector<double>>(numNodesIn, std::vector<double>(numNodesOut, 0.0));
        costGradientBias = std::vector<double>(numNodesOut, 0.0);

        // initialize weights and biases with 0
        weights = std::vector<std::vector<double>>(numNodesIn, std::vector<double>(numNodesOut, 0.0));
        biases = std::vector<double>(numNodesOut, 0.0);

        InitializeRandomWeights();
    }

    void InitializeRandomWeights()
    {
        srand(static_cast<unsigned int>(time(0)));
        for (int nodeIn = 0; nodeIn < numNodesIn; ++nodeIn)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; ++nodeOut)
            {
                double randomValue = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
                weights[nodeIn][nodeOut] = randomValue / sqrt(numNodesIn);
            }
        }
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

    double NodeCost(double outputActivation, double expectedOutput)
    {
        double error = outputActivation - expectedOutput;
        return error * error;
    }

    void ApplyGradients(double learnRate)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientBias[nodeOut] * learnRate;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn][nodeOut] -= costGradientWeights[nodeIn][nodeOut] * learnRate;
            }
        }
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

    // TODO: edit input variable
    // take in a vector of inputs
    double Loss(const std::vector<double> &inputs, const std::vector<double> &expectedOutputs)
    {
        std::vector<double> outputs = feedForward(inputs);
        Layer &outputLayer = layers[layers.size() - 1];
        double cost = 0;
        for (size_t nodeOut = 0; nodeOut < outputs.size(); nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], expectedOutputs[nodeOut]);
        }
        return cost;
    }

    // loss function, takes in set of inputs and a list of expected outputs
    double Loss(const std::vector<std::vector<double>> &inputsList, const std::vector<std::vector<double>> &expectedOutputsList)
    {
        double totalCost = 0.0;

        for (size_t i = 0; i < inputsList.size(); ++i)
        {
            const std::vector<double> &inputs = inputsList[i];
            const std::vector<double> &expectedOutputs = expectedOutputsList[i];

            // Compute cost for the current data point and add to total cost
            double dataPointCost = Loss(inputs, expectedOutputs);
            totalCost += dataPointCost;
        }
        return totalCost / inputsList.size();
    }

    void Learn(const std::vector<std::vector<double>> &inputsList, const std::vector<std::vector<double>> &expectedOutputsList, double learnRate)
    {
        const double h = 0.0001;
        double originalCost = Loss(inputsList, expectedOutputsList);

        for (Layer &layer : layers)
        {
            for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
            {
                for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
                {
                    layer.weights[nodeIn][nodeOut] += h;
                    double deltaCost = Loss(inputsList, expectedOutputsList) - originalCost;
                    layer.weights[nodeIn][nodeOut] -= h;
                    layer.costGradientWeights[nodeIn][nodeOut] = deltaCost / h;
                }
            }

            for (int biasIndex = 0; biasIndex < layer.biases.size(); biasIndex++)
            {
                layer.biases[biasIndex] += h;
                double deltaCost = Loss(inputsList, expectedOutputsList) - originalCost;
                layer.biases[biasIndex] -= h;
                layer.costGradientBias[biasIndex] = deltaCost / h;
            }
        }

        ApplyGradients(learnRate);
    }

    void ApplyGradients(double learnRate)
    {
        for (Layer &layer : layers)
        {
            layer.ApplyGradients(learnRate);
        }
    }
};

int main()
{
    NeuralNetwork network({2, 3, 2});

    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> expectedOutputs = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}};

    int epochs = 1000;
    double learningRate = 0.1;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        network.Learn(inputs, expectedOutputs, learningRate);

        double cost = network.Loss(inputs, expectedOutputs);

        if (epoch % 100 == 0 || epoch == epochs - 1)
        {
            std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
        }
    }

    std::cout << "\nTesting the trained network:\n";
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        int classification = network.Classify(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] ";
        std::cout << "Classified as: " << classification;
        std::cout << " (Expected: " << (expectedOutputs[i][0] > expectedOutputs[i][1] ? 0 : 1) << ")\n";
    }
}