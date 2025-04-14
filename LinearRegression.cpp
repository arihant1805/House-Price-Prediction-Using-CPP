#include <bits/stdc++.h>
using namespace std;

#define double long double



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

    void fit(const vector<vector<double>>& X, const vector<double>& y, double learning_rate = 5e-7, bool regularization = false, double lambda = 0.01) {
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
    double evaluate(const vector<vector<double>>& X, const vector<double>& y) {
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
    vector<double> predict(const vector<vector<double>>& x) {
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

};

pair<pair<vector<vector<double>>,vector<double>>,pair<vector<vector<double>>,vector<double>>> train_test_split(
    const vector<vector<double>>& X, 
    const vector<double>& y, 
    double test_size = 0.2) 
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