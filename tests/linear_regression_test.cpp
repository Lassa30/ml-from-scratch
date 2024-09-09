#include <iomanip>
#include <models/linear_regression.hpp>
#include <utils/matrix.hpp>
#include <utils/utils.hpp>

int main() {

  std::vector<double> data;
  mlfs::utils::dataFromCsv(data, "../tests/winequality-white.csv", ';');

  std::cout << std::fixed << std::setprecision(5);

  std::vector<double> X;
  std::vector<double> y;

  for (auto i = 0; i < data.size(); i++) {
    if (i % 12 != 11)
      X.push_back(data[i]);
    else
      y.push_back(data[i]);
  }

  mlfs::Matrix designMatrix(4898, 11, X);
  mlfs::Matrix targetColumn(4898, 1, y);

  mlfs::LinearRegressionSGD LinReg(256, 1e-5, 1000);

  LinReg.train(designMatrix, targetColumn, 42);

  auto prediction = LinReg.predict(designMatrix);

  std::cout << "\nSCORE:\n\t";
  std::cout << LinReg.score(prediction, targetColumn) << std::endl;
}