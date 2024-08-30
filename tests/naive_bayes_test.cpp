#include "../include/models/naive_bayes.hpp"
#include "../include/utils/matrix.hpp"

int main() {
    mlfs::Matrix iris_features_train = mlfs::Matrix(3, 4, {5.1, 3.5, 1.4, 0.2, 
                                                           6.1, 2.8, 4.7, 1.2, 
                                                           6.4, 3.0 ,5.6, 2.2}
    );
    mlfs::Matrix iris_target_train = mlfs::Matrix({0, 1, 2}, mlfs::Matrix::AXIS::COLUMN);

    mlfs::Matrix iris_features_test = mlfs::Matrix(3, 4, {5.1, 3.5, 1.4, 0.2, 
                                                           6.1, 2.8, 4.7, 1.2, 
                                                           6.4, 3.0 ,5.6, 2.2}
    );
    mlfs::Matrix iris_target_test = mlfs::Matrix({0, 1, 2}, mlfs::Matrix::AXIS::COLUMN);

    auto nb = mlfs::GaussianNaiveBayes();
    nb.train(iris_features_train, iris_target_train);
    
    auto res = nb.predict(iris_features_test);

    res.print_matrix();

    iris_target_test.print_matrix();
}

/* 
setosa:
5.1, 3.5, 1.4, 0.2

4.6, 3.4, 1.4, 0.3
versicolor:
6.1, 2.8, 4.7, 1.2

6.4, 2.9, 4.3, 1.4
virginica:
6.4, 2.8 ,5.6, 2.2

6.7, 3.0, 5.2, 2.3


{4.6, 3.4, 1.4, 0.3,
6.4, 2.9, 4.3, 1.4,
6.7, 3.0, 5.2, 2.3}
*/