#ifndef DataPreprocessing_h
#define DataPreprocessing_h

#include <bits/stdc++.h>
using namespace std;
#define double long double

pair<vector<vector<double>>,vector<double>> readData(const string& filename);

pair<pair<vector<vector<double>>,vector<double>>,pair<vector<vector<double>>,vector<double>>> train_test_split(
    const vector<vector<double>>& X, 
    const vector<double>& y, 
    double test_size = 0.2);

#endif