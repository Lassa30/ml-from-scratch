# The Purpose of This Project

I enjoy working with C++ and have recently started my ML journey. To combine both interests, I decided to create this project.

This project is my way of learning. By implementing ML algorithms from scratch in C++, I can:

- Improve my basic programming skills.
- Understand how these algorithms really work step by step.

## Project tree

```tree
├── CMakeLists.txt
├── examples
│   ├──data
|
|   # headers
├── include 
│   ├── models
│   └── utils
|
|   # implementation
├── src 
│   ├── models
│   │   ├── linear_regression.cpp
│   │   ├── ...
│   └── utils
```

## Implemented models

### [K-Nearest-Neighbors Classifier](https://github.com/Lassa30/ml-from-scratch/blob/main/src/models/neighbors.hpp)

### [Gaussian Naive Bayes Classifier](https://github.com/Lassa30/ml-from-scratch/blob/main/src/models/naive_bayes.cpp)

### [Linear Regression](https://github.com/Lassa30/ml-from-scratch/blob/main/src/models/linear_regression.cpp)

## TODO

- [ ] Change interfaces: remove train member function with "Trainer" class for models which have relations to Optimizer and LossFunction.

- [ ] Refactor LinearRegression and implement LogisticRegression using new interfaces.

- [ ] Create a playground webpage to demonstrate how implemented models work.
