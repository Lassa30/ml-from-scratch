#include <models/linear_regression.hpp>

#include <utils/matrix.hpp>
#include <utils/optimizer.hpp>
#include <utils/utils.hpp>

#include <memory>

#include <iomanip>

int main() {
  std::cout << std::fixed;

  std::vector<double> data;
  mlfs::utils::dataFromCsv(data, "../tests/winequality-white.csv", ';');

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

  // Construct linear regression
  auto opt = std::make_unique<mlfs::optim::SGD>(3.5e-5);
  auto loss = std::make_unique<mlfs::optim::MSE>();
  std::cout << "lr: " << opt->getLearningRate() << '\n';

  mlfs::LinearRegression LinReg(std::move(opt), std::move(loss));

  LinReg.train(designMatrix, targetColumn, 1024, 1000, -100, 100, 69);

  auto prediction = LinReg.predict(designMatrix);

  std::cout << std::setprecision(3);
  std::cout << "Compare prediction to target:\n";
  for (int elem = 0; elem < 10; ++elem)
    std::cout << prediction.get(elem, 1) << " | " << targetColumn.get(elem, 1) << '\n';

  std::cout << "SCORE:\n\t";
  std::cout << LinReg.score(prediction, targetColumn) << std::endl;
}