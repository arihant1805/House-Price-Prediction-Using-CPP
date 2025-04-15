#ifndef LinearRegression_h
#define LinearRegression_h

#include <bits/stdc++.h>
using namespace std;
#define double long double

class LinearRegression {
    public:
        vector<double> W;
        double b;
    
        LinearRegression(int n_features) {
            b = 0.0;
            for (int i = 0; i < n_features; ++i) {
                W.push_back(0.0);
            }
        }
    
        void fit(const vector<vector<double>>& X, const vector<double>& y, double learning_rate = 5e-7, bool regularization = false, double lambda = 0.01);
        vector<double> predict(const vector<vector<double>>& x);

        double evaluate(const vector<vector<double>>& X, const vector<double>& y);
};

#endif