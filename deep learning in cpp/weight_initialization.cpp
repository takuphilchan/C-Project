#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

// function to initialize weights using He (Kaiming) Initialization
vector<double> initializeWeightsHe(int n_in, int n_out) {
    // vector to store initialized weights
    vector<double> weights(n_in * n_out); 

    // compute the standard deviation for normal distribution
    double stddev = sqrt(2.0 / n_in);

    // use the <random> library to generate random numbers with a normal distribution
    default_random_engine generator; 
    normal_distribution<double> distribution(0.0, stddev);

    // initialize weights with random values from the normal distribution
    for (int i = 0; i < n_in * n_out; ++i) {
        weights[i] = distribution(generator);
    }
    return weights;
}
// function to initialize weights using Xavier (Glorot) Initialization
vector<double> initializeWeightsXavier(int n_in, int n_out) {
    // vector to store the initialized weights
    vector<double> weights(n_in * n_out);

    // compute the range limit for uniform distribution
    double limit = sqrt(6.0 / (n_in + n_out));

    // seed the random number generator
    srand(static_cast<unsigned int>(time(0)));
    
    // initialize weights with random values in the range [-limit, limit]
    for (int i = 0; i < n_in * n_out; ++i) {
        weights[i] = -limit + static_cast<double>(rand()) / RAND_MAX * (2 * limit);
    }
    return weights;
}

int main() {
    // number of input and output neurons
    int n_in = 4;
    int n_out = 6;

    // initialize weights using He initialization
    auto weights = initializeWeightsHe(n_in, n_out);
    cout << "He Initialized Weights:\n";
    for (const auto& weight : weights) {
        cout << weight << " ";
    }
    cout << endl; 
    return 0; 
}