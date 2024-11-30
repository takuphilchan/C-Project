#include <iostream>
#include <vector>
#include <memory>

using namespace std;

// Define the Var class
class Var : public enable_shared_from_this<Var> {
public:
    double value; // The value of the variable
    double grad;  // The gradient of the variable

    // Constructor to initialize value and set gradient to 0
    Var(double v) : value(v), grad(0.0) {}

    // Structure to represent edges in the computational graph
    struct Edge {
        shared_ptr<Var> parent; // Parent variable
        double weight;          // Contribution to the gradient
    };

    // List of parents in the computational graph
    vector<Edge> parents;

    // Backward propagation for reverse-mode AD
    void backward() {
        this->grad = 1.0; // Seed gradient for the output variable
        backward_recursive();
    }

private:
    void backward_recursive() {
        // Propagate gradients to parents
        for (auto& edge : parents) {
            edge.parent->grad += edge.weight * this->grad; // Chain rule
            edge.parent->backward_recursive();            // Recursive call
        }
    }
};

// Overload the addition operator for shared_ptr<Var>
shared_ptr<Var> operator+(const shared_ptr<Var>& lhs, const shared_ptr<Var>& rhs) {
    auto result = make_shared<Var>(lhs->value + rhs->value);
    result->parents.push_back({lhs, 1.0}); // d(result)/d(lhs) = 1
    result->parents.push_back({rhs, 1.0}); // d(result)/d(rhs) = 1
    return result;
}

// Overload the multiplication operator for shared_ptr<Var>
shared_ptr<Var> operator*(const shared_ptr<Var>& lhs, const shared_ptr<Var>& rhs) {
    auto result = make_shared<Var>(lhs->value * rhs->value);
    result->parents.push_back({lhs, rhs->value}); // d(result)/d(lhs) = rhs->value
    result->parents.push_back({rhs, lhs->value}); // d(result)/d(rhs) = lhs->value
    return result;
}

int main() {
    auto x = make_shared<Var>(2.0);
    auto y = make_shared<Var>(3.0);

    auto z = (x * y) + x;

    // perform backward pass
    z->backward();

    cout << "Value of z: " << z->value << endl;
    cout << "Gradient dz/dx: " << x->grad << endl;
    cout << "Gradient dz/dy: " << y->grad << endl; 
    return 0;
}