#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double getRandom() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen);
}

class Layer {
public:
    vector<double> neurons;
    vector<vector<double>> weights;
    vector<double> bias;

    Layer(int inputSize, int outputSize) {
        neurons.resize(outputSize);
        bias.resize(outputSize); 
        weights.resize(inputSize, vector<double>(outputSize));

        for (int i=0; i < inputSize; ++i)
            for (int j=0; j < outputSize; ++j)
                weights[i][j] = getRandom();
        
        for (double &b : bias)
            b = getRandom(); 

    } 

};

class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<int> &layerSizes) {
        for (size_t i=1; i<layerSizes.size(); ++i) {
            layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
        }
    }

    vector<double> forward(const vector<double> &input) {
        vector<double> activation = input;

        for (Layer &layer : layers) {
            vector<double> newActivation(layer.neurons.size());

            for (size_t j = 0; j < layer.neurons.size(); ++j) {
                double sum = layer.bias[j];
                for (size_t i = 0; i < activation.size(); ++i)
                    sum += activation[i] * layer.weights[i][j];
                newActivation[j] = sigmoid(sum);
            }

            activation = newActivation;
        }
        return activation;
    }

    void backpropagrate(const vector<double> &input, const vector<double> &target, double learningRate) {
        vector<vector<double>> activations;
        activations.push_back(input); 

        for (Layer &layer : layers) {
            vector<double> newActivation(layer.neurons.size());
            for (size_t j = 0; j < layer.neurons.size(); ++j) {
                double sum = layer.bias[j];
                for (size_t i = 0; i < activations.back().size(); ++i)
                    sum += activations.back()[i] * layer.weights[i][j];
                newActivation[j] = sigmoid(sum);
            }
        activations.push_back(newActivation);
        }

        vector<double> delta = activations.back();
        for (size_t i = 0; i < delta.size(); ++i)
            delta[i] = (delta[i] - target[i]) * sigmoidDerivative(delta[i]);

        for (int l = layers.size() - 1; l >= 0; --l) {
            vector<double> newDelta(layers[l].weights.size());

            for (size_t i = 0; i < layers[l].weights.size(); ++i){
                double error = 0.0;
                for (size_t j = 0; j < layers[l].neurons.size(); ++j){
                    error += delta[j] * layers[l].weights[i][j];
                    layers[l].weights[i][j] -= learningRate * activations[l][i] * delta[j];

                }
                newDelta[i] = error * sigmoidDerivative(activations[l][i]);
            }
            for (size_t j = 0; j < layers[l].neurons.size(); ++j)
                layers[l].bias[j] -=learningRate * delta[j];

            delta = newDelta;
        }
        
    }

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalError = 0.0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                vector<double> output = forward(inputs[i]);
                for (size_t j = 0; j < targets[i].size(); ++j){
                    totalError += pow(output[j] - targets[i][j], 2);
                }
                backpropagrate(inputs[i], targets[i], learningRate);
            }
            if (epoch % 100 == 0)
                cout << "Epoch" << epoch << " - Error: " << totalError / inputs.size() << endl;
        }
    }

};


int main() {
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1} 
    };

    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    NeuralNetwork nn({2, 2, 1}); 

    nn.train(inputs, targets, 5000, 0.1);
    
    for ( const auto &input : inputs) {
        vector<double> output = nn.forward(input);
        cout<< "Input: {"<< input[0] <<", "<< input[1] << "} - Output:" << output[0]<< endl; 
    }
    return 0;
}



