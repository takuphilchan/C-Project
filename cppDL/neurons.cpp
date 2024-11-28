#include <iostream>
#include <vector>

using namespace std; 

// ReLU Activation Function
double relu(double x) {
    return x > 0 ? x : 0;
}
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Sigmoid Activation Function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

