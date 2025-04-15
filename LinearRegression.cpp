#include <bits/stdc++.h>
using namespace std;
#define double long double
#include "LinearRegression.h"


void LinearRegression :: fit(const vector<vector<double>>& X, const vector<double>& y, double learning_rate, bool regularization, double lambda) {
        int m = X.size();
        int n = X[0].size();
        double alpha = learning_rate;
        int epochs = 1000;
        double costx = 0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            vector<double> y_pred(m);
            for (int i = 0; i < m; ++i) {
                y_pred[i] = b;
                for (int j = 0; j < n; ++j) {
                    y_pred[i] += W[j] * X[i][j];
                }
            }

            for (int i = 0; i < m; ++i) {
                double error = y_pred[i] - y[i];
                b -= alpha * error / (2 * m);
                for (int j = 0; j < n; ++j) {
                    W[j] -= alpha * error * X[i][j] / m;
                    if (regularization) {
                        W[j] -= alpha * lambda * W[j] / m;
                    }
                }
            }
            if(epoch % 50 == 0) {
                double cost = 0.0;
                for (int i = 0; i < m; ++i) {
                    cost += pow(y_pred[i] - y[i], 2);
                }
                cost /= (2 * m);
                cout << "Epoch " << epoch + 1 << ", Cost: " << cost << endl;
                if (regularization) {
                    for (int j = 0; j < n; ++j) {
                        cost += lambda * W[j] * W[j] / (2 * m);
                    }
                }
                if (abs(cost - costx) < 1e-6) {
                    cout << "Converged at epoch " << epoch + 1 << endl;
                    break;
                }
                costx = cost;
            }
        }
    }
double LinearRegression :: evaluate(const vector<vector<double>>& X, const vector<double>& y) {
        int m = X.size();
        double total_error = 0.0;
        for (int i = 0; i < m; ++i) {
            double y_pred = b;
            for (int j = 0; j < X[0].size(); ++j) {
                y_pred += W[j] * X[i][j];
            }
            total_error += pow(y_pred - y[i], 2);
        }
        return sqrt(total_error / m);
    }
vector<double> LinearRegression :: predict(const vector<vector<double>>& x) {
        int m = x.size();
        vector<double> y_pred(m);
        for (int i = 0; i < m; ++i) {
            y_pred[i] = b;
            for (int j = 0; j < x[0].size(); ++j) {
                y_pred[i] += W[j] * x[i][j];
            }
        }
        return y_pred;
    }
