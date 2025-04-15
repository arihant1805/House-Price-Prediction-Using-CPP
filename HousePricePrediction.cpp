#include <bits/stdc++.h>
using namespace std;
#include "DataPreprocessing.h"
#include "LinearRegression.h"
#define double long double

int main(){
    string filename = "data_cleaned.csv";
    auto [X, y] = readData(filename);
    cout << "Data loaded successfully." << endl;

    cout << "Number of samples: " << X.size() << endl;
    cout << "Number of features: " << X[0].size() << endl;
    cout << "First 5 samples:" << endl;
    for (int i = 0; i < 5; ++i) {
        cout << "X: ";
        for (double val : X[i]) {
            cout << val << " ";
        }
        cout << "y: " << y[i] << endl;
    }

    cout << "Splitting data into training and testing sets..." << endl;
    cout << "Training set size: " << X.size() * 0.8 << endl;
    cout << "Testing set size: " << X.size() * 0.2 << endl;
    auto [train, test] = train_test_split(X, y, 0.2);
    auto [X_train, y_train] = train;
    auto [X_test, y_test] = test;


    LinearRegression model(X_train[0].size());
    

    cout << "Before training\nModel weights: ";
    for (int i = 0; i < model.W.size(); ++i) {
        cout << model.W[i] << " ";
    }

    cout << " Bias: " << model.b << endl;
    

    cout << "Training the model..." << endl;
    model.fit(X_train, y_train, 5e-5, true, 0.01);
    

    cout << "After training\nModel weights: ";
    for (int i = 0; i < model.W.size(); ++i) {
        cout << model.W[i] << " ";
    }

    cout << " Bias: " << model.b << endl;


    double rms = model.evaluate(X_test, y_test);
    cout << "RMS on test data : " << rms << endl;

    return 0;
}