#include <eigen3/Eigen/Dense>
#include <iostream>
#include <models/naive_bayes.hpp>

using namespace mlfs;

int main() {
  // Toy data
  MatrixXd features(4, 4);
  MatrixXd target(4, 1);

  features << 500, 200, 500, 300, -100, -300, -100, -200, 100, 250, 590, 350,
      -300, -250, -400, -599;
  target << 1, 0, 1, 0;

  std::cout << "example features:\n" << features << std::endl;
  std::cout << "example target:\n" << target << std::endl;

  // Create a model object
  GaussianNaiveBayes nb;

  // Train it
  nb.train(features, target);

  // Make a prediction
  auto probas = nb.predict_proba(features);
  std::cout << "the probas are:\n" << probas << std::endl;
  auto prediction = nb.predict(features);
  std::cout << "the predictions are:\n" << prediction << std::endl;
}