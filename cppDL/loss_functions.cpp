#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std; 

double categoricalCrossEntropy(const vector<vector<double>>& y_pred, const vector<vector<double>>& y_true) {
    double loss = 0.0;
    size_t batch_size = y_pred.size();

    for (size_t i = 0; i < batch_size; ++i) {
        // ensure y_pred and y_true dimensions match
        if (y_pred[i].size() != y_true[i].size()) {
            cerr << "Error: Mismatched dimensions between predicted and true" << endl;
            return -1.0;
        }

        for (size_t j = 0; j < y_pred[i].size(); ++j) {
            // clamp the predictions to avoid log(0)
            double p = clamp(y_pred[i][j], 1e-15, 1.0 - 1e-15);
            loss += y_true[i][j] * log(p);
        }
    }
    return -loss / batch_size;
}

double meanSquaredError(const vector<vector<double>>& y_pred, const vector<vector<double>>& y_true) {
    double loss = 0.0;
    size_t batch_size = y_pred.size();

    for (size_t i = 0; i < batch_size; ++i) {
        // ensure y_pred and y_true dimensions match
        if (y_pred[i].size() != y_true[i].size()) {
            cerr << "Error: Mismatched dimensions between predicted and true" << endl;
            return -1.0;
        }

        for (size_t j = 0; j < y_pred[i].size(); ++j) {
            // compute the squared differences
            loss += pow(y_pred[i][j] - y_true[i][j], 2);
        }
    }
    return loss / (batch_size * y_pred[0].size());
}

int main() {
    vector<vector<double>> y_pred = {
        {0.0141, 0.0384, 0.0032, 0.9443},
        {0.1, 0.6, 0.1, 0.2}
    };
    vector<vector<double>> y_true_cce = {
        {0, 0, 0, 1},
        {0, 1, 0, 0}  
        };

    vector<vector<double>> y_true_mse = {
        {0.01, 0.04, 0.03, 0.94}, 
        {0.1, 0.5, 0.1, 0.2}      
    };

    double MSEloss = meanSquaredError(y_pred, y_true_mse);
    cout<< "Mean Squared Error Loss: " << fixed << setprecision(4) << MSEloss << endl; 

    double CCEloss = categoricalCrossEntropy(y_pred, y_true_cce);
    cout<< "Categorical Cross-Entropy Loss: " << fixed << setprecision(4) << CCEloss << endl;

    return 0;

}
