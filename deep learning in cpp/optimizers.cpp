#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// define stochastic gradient descent (sgd) optimizer
void sgd(vector<double>& weights, const vector<double>& gradients, double learning_rate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}

// define adam optimizer
void adam(vector<double>& weights, const vector<double>& gradients, vector<double>& m, 
          vector<double>& v, double learning_rate, double beta1, double beta2, 
          double epsilon, int t) {
    for (size_t i = 0; i < weights.size(); ++i) {
     // update biased first moment estimate
     m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
     // update biased second moment estimate 
     v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] * gradients[i]);
    // compute bias-corrected moment estimates
    double m_hat = m[i] / (1 - pow(beta1, t));
    double v_hat = v[i] / (1 - pow(beta2, t));
    // update weights 
    weights[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}
int main() {
    // define weights and gradients
    vector<double> weights = {0.3, -0.3, 0.8}; 
    vector<double> gradients = {0.2, -0.1, 0.07};
    vector<double> m(weights.size(), 0.0); // initialize first moment vector
    vector<double> v(weights.size(), 0.0); // initialize second moment vector
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    int t = 1; // time step
    double learning_rate = 0.01; 

    cout << "Before update: ";
    for (double w : weights) cout << w << " ";
    cout << endl;
    // sgd(weights, gradients, learning_rate);
    adam(weights, gradients, m, v, learning_rate, beta1, beta2, epsilon, t);
    cout << "After update: ";
    for (double w : weights) cout << w << " ";
    cout << endl;

    return 0;
}   




