# House Price Prediction

This project implements a house price prediction system using a custom Linear Regression model in C++ and a data cleaning pipeline in Python.

## Project Overview

- **Data Cleaning:**  
  The `clean.py` script processes the raw dataset (`data.csv`) by selecting relevant features, removing missing values and duplicates, and saving the result as `data_cleaned.csv`.

- **Model Training:**  
  The `LinearRegression.cpp` file contains the C++ implementation of the Linear Regression algorithm. It reads the cleaned dataset, applies a standard scalar (centering each feature and target variable by subtracting the mean and scaling by the standard deviation), trains the model, and evaluates predictions.

## Project Structure

- `data.csv`           : Raw dataset with housing information.
- `clean.py`           : Python script for data preprocessing.
- `data_cleaned.csv`   : Clean dataset generated from `clean.py`.
- `LinearRegression.cpp`: C++ source code implementing the Linear Regression model.
- `README.md`          : Project documentation.

## Requirements

### Python

- Python 3.x with the `pandas` library.
- Install dependencies with:
  ```bash
  pip install pandas
  ```

### C++

- A C++ compiler supporting C++11 or later (e.g., g++).
- Standard C++ libraries.

## Build and Run Instructions

### 1. Data Cleaning

Run the following command to clean the dataset:
```bash
python3 clean.py
```
This generates `data_cleaned.csv` which is required by the C++ model.

### 2. Compile and Run the C++ Model

Compile the C++ source code using:
```bash
g++ -o LinearRegression LinearRegression.cpp -O2
```
Then execute the model with:
```bash
./LinearRegression
```

## Notes

- Ensure that `data_cleaned.csv` is available before running the C++ application.
- Adjust the parameters in the source files as needed.
- For any issues during compilation or execution, verify that all dependencies are properly installed.

## License
This project is licensed under the [Apache License 2.0](LICENSE).
