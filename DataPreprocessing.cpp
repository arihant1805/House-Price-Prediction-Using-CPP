#include <bits/stdc++.h>
using namespace std;
#define double long double
#include "DataPreprocessing.h"

pair<vector<vector<double>>,vector<double>> readData(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> X;
    vector<double> y;
    string line;
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        y.push_back(row.front());
        row.erase(row.begin());
        X.push_back(row);
    }

    int m = X.size();
    int n = X[0].size();

    // Apply standard scalar to features: subtract mean and divide by standard deviation for each feature.
    for(int i = 0; i < n; i++){
        double mean = 0.0;
        for(int j = 0; j < m; j++) mean += X[j][i];

        mean /= m;

        for(int j = 0; j < m; j++) X[j][i] -= mean;
        
        double stddev = 0.0;
        for(int j = 0; j < m; j++) stddev += X[j][i] * X[j][i];
        
        stddev = sqrt(stddev / m);

        for(int j = 0; j < m; j++) X[j][i] /= stddev;  
    }

    // Apply standard scalar to target variable y.
    double mean = 0.0;
    for(int j = 0; j < m; j++) mean += y[j];
    
    mean /= m;

    for(int j = 0; j < m; j++) y[j] -= mean;

    double stddev = 0.0;

    for(int j = 0; j < m; j++) stddev += y[j] * y[j];
    
    stddev = sqrt(stddev / m);

    for(int j = 0; j < m; j++)
        y[j] /= stddev;

    return {X, y};
}



pair<pair<vector<vector<double>>,vector<double>>,pair<vector<vector<double>>,vector<double>>> train_test_split(
    const vector<vector<double>>& X, 
    const vector<double>& y, 
    double test_size) 
    {
    int m = X.size();
    int test_size_int = m * test_size;
    vector<vector<double>> X_train;
    vector<vector<double>> X_test;
    vector<double> y_train;
    vector<double> y_test;
    for (int i = 0; i < m; ++i) {
        if (i < m - test_size_int) {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        } else {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
    }

    return {{X_train, y_train}, {X_test, y_test}};
}