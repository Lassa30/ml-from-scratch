#include <models/naive_bayes.hpp>
#include <utils/matrix.hpp>

#include <random>

int main() {
  // random config

  std::srand(42);
  const int size = 1000;
  auto randomNoise = std::vector<double>(size);
  for (auto &randomNumber : randomNoise) {
    auto num = std::rand();
    randomNumber = (num % 2 == 0) ? std::rand() * 1.0 / RAND_MAX
                                  : std::rand() * -1.0 / RAND_MAX;
  }

  auto randomNoiseSum =
      std::accumulate(randomNoise.begin(), randomNoise.end(), 0.0) / size;
  std::cout << "RandomNoise: " << randomNoiseSum << '\n';

  // Matrix init
  mlfs::Matrix irisFeaturesTrain = mlfs::Matrix(
      3, 4, {5.1, 3.5, 1.4, 0.2, 6.1, 2.8, 4.7, 1.2, 6.4, 3.0, 5.6, 2.2});
  mlfs::Matrix irisTargetTrain =
      mlfs::Matrix({0, 1, 2}, mlfs::Matrix::AXIS::COLUMN);

  std::cout << "Train design matrix:\n";
  irisFeaturesTrain.print_matrix();
  std::cout << std::endl;

  mlfs::Matrix irisFeaturesTest = irisFeaturesTrain;

  irisFeaturesTest += randomNoiseSum;

  mlfs::Matrix iris_target_test =
      mlfs::Matrix({0, 1, 2}, mlfs::Matrix::AXIS::COLUMN);

  // Train
  auto nb = mlfs::GaussianNaiveBayes();
  nb.train(irisFeaturesTrain, irisTargetTrain);

  // Test
  auto prediction = nb.predict(irisFeaturesTest);

  std::cout << "Test design matrix:\n";
  irisFeaturesTest.print_matrix();
  std::cout << std::endl;

  std::cout << "Test prediction:\n";
  prediction.print_matrix();
  std::cout << std::endl;

  std::cout << "Test target:\n";
  iris_target_test.print_matrix();
  std::cout << std::endl;
}
