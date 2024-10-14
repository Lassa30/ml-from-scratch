#include <utils/matrix.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>

namespace mlfs {

bool close_to(const double &lhs, const double &rhs) { return std::abs(lhs - rhs) < EPSILON; }

// private
std::string Matrix::shape_err_msg(const std::string &method, const Matrix &rhs) const {
  std::stringstream err_msg;

  err_msg << method << '\n';
  err_msg << "The shapes don't fit\n";

  err_msg << "lhs: (" << shape().first << " ," << shape().second << ")\n";

  err_msg << "rhs: (" << rhs.shape().first << " ," << rhs.shape().second << ")\n";

  return err_msg.str();
}

void Matrix::swap(Matrix &rhs) noexcept {
  std::swap(rows_, rhs.rows_);
  std::swap(cols_, rhs.cols_);
  std::swap(data_, rhs.data_);
}

// public
Matrix::Matrix(const std::size_t &rows, const std::size_t &columns) : cols_{columns}, rows_{rows} {
  data_ = std::vector<double>(size(), 0.0);
}

Matrix::Matrix(const std::size_t &rows, const std::size_t &columns, const std::vector<double> &rhs)
    : rows_{rows}, cols_{columns}, data_(rows * columns, 0.0) // Initialize data_ with size rows*columns
{
  if (size() == rhs.size()) {
    // Directly copy elements into data_
    std::copy(rhs.begin(), rhs.end(), data_.begin());
  } else if (size() < rhs.size()) {
    // Copy only the needed elements
    std::copy(rhs.begin(), rhs.begin() + size(), data_.begin());
  } else {
    throw std::logic_error("Matrix(const std::size_t &rows, const std::size_t &columns, const "
                           "std::vector<double> &rhs):\n\tstd::vector size is less than "
                           "rows*columns\n");
  }
}

Matrix::Matrix(const std::vector<double> &rhs, AXIS axis) {
  if (rhs.empty()) {
    throw std::logic_error("Couldn't construct a Matrix from an empty std::vector<double>\n");
  }
  if (axis != AXIS::ROW && axis != AXIS::COLUMN) {
    throw std::logic_error("Invalid axis. Use AXIS::ROW or AXIS::COLUMN.");
  }

  rows_ = (axis == AXIS::ROW) ? 1 : rhs.size();
  cols_ = (axis == AXIS::ROW) ? rhs.size() : 1;

  data_ = std::vector<double>(rhs);
}

Matrix::Matrix(Matrix &&rhs) noexcept : rows_{rhs.rows_}, cols_{rhs.cols_}, data_{std::move(rhs.data_)} {
  rhs.rows_ = 0;
  rhs.cols_ = 0;
  rhs.data_ = {};
}

Matrix &Matrix::operator=(const Matrix &rhs) {
  // copy 'n' swap idiom
  if (this != &rhs) {
    Matrix cp(rhs);
    swap(cp);
  }
  return *this;
}

Matrix &Matrix::operator=(Matrix &&rhs) noexcept {
  if (this != &rhs) {
    swap(rhs);
  }
  return *this;
}

void Matrix::reshape(const std::size_t &cols, const std::size_t &rows) {
  if (cols * rows == 0) {
    throw std::logic_error("LogicError: Reshaping to zero sized Matrix is impossible.\n");
  }

  if (cols * rows == size()) {
    cols_ = cols;
    rows_ = rows;
  }
}

Matrix &Matrix::operator+=(const Matrix &rhs) {
  if (!(rows_ == rhs.rows_ && cols_ == rhs.cols_)) {
    throw std::logic_error(shape_err_msg("Matrix & operator+=(const Matrix & rhs)", rhs));
  }

  for (auto i = 0; i < size(); ++i) {
    data_[i] += rhs.data_[i];
  }

  return *this;
}

Matrix &Matrix::operator+=(const double &rhs) {
  for (auto i = 0; i < size(); ++i) {
    data_[i] += rhs;
  }

  return *this;
}

Matrix &Matrix::operator-=(const Matrix &rhs) {
  if (!(rows_ == rhs.rows_ && cols_ == rhs.cols_)) {
    throw std::logic_error(shape_err_msg("Matrix & operator-=(const Matrix &rhs)", rhs));
  }

  for (auto i = 0; i < size(); ++i) {
    data_[i] -= rhs.data_[i];
  }

  return *this;
}

Matrix &Matrix::operator*=(const double &rhs) {
  for (auto &elem : data_)
    elem *= rhs;

  return *this;
}

Matrix &Matrix::operator/=(const double &rhs) {
  if (close_to(rhs, 0.0))
    throw std::runtime_error("Zero division. Matrix / 0.0 is not defined");

  for (auto &elem : data_)
    elem /= rhs;

  return *this;
}

Matrix Matrix::operator+(const double &rhs) const {
  Matrix temp(*this);
  temp += rhs;
  return temp;
}

Matrix Matrix::operator+(const Matrix &rhs) const {
  Matrix temp(*this);
  temp += rhs;
  return temp;
}

Matrix Matrix::operator-(const Matrix &rhs) const {
  Matrix temp(*this);
  temp -= rhs;
  return temp;
}

Matrix Matrix::operator-() const {
  auto temp(*this);
  std::size_t i = 0;

  for (auto &elem : temp.data_)
    elem = -elem;

  return temp;
}

Matrix Matrix::operator*(const double &rhs) const {
  Matrix temp(*this);
  temp *= rhs;
  return temp;
}

Matrix Matrix::operator*(const Matrix &rhs) const {
  if (!(cols_ == rhs.cols_ && rows_ == rhs.rows_)) {
    throw std::logic_error("Matrices should have equal shapes.\n");
  }

  std::vector<double> cp(size(), 1.0);
  for (auto i = 0; i < size(); i++) {
    cp[i] *= data_[i] * rhs.data_[i];
  }

  return Matrix(rows_, cols_, cp);
}

Matrix Matrix::operator/(const double &rhs) const {
  if (close_to(rhs, 0.0)) {
    throw std::runtime_error("Zero division.\n"
                             "Method: Matrix operator/(const double & rhs)\n"
                             "rhs=" +
                             std::to_string(rhs) + '\n');
  }

  return *this * (1.0 / rhs);
}

// Naive algorithm works slow:
// Strassen algorithm is going to be used.

Matrix Matrix::matmul(const Matrix &rhs) const {

  if (cols_ != rhs.rows_) {
    throw std::logic_error(shape_err_msg("Matrix matmul(const Matrix & rhs)", rhs));
  }

  Matrix product(rows_, rhs.cols_, std::vector<double>(rows_ * rhs.cols_, 0.0));

  // Naive algprintorithm is better for small matrices.
  // source: wikipedia

  for (std::size_t i = 0; i < rows_; ++i) {
    for (std::size_t j = 0; j < rhs.cols_; ++j) {
      for (std::size_t k = 0; k < cols_; ++k) {
        product.data_.at(i * rhs.cols_ + j) += (data_.at(i * cols_ + k) * rhs.data_.at(k * rhs.cols_ + j));
      }
    }
  }

  return product;
}

Matrix Matrix::sum(AXIS axis) {
  if (axis == AXIS::ROW || axis == AXIS::COLUMN) {
    std::size_t row_or_col = (axis == AXIS::ROW) ? rows_ : cols_;

    std::vector<double> result(row_or_col);

    for (auto i = 0; i < size(); ++i) {
      result[i / row_or_col] += data_[i];
    }

    return Matrix(result, axis);
  }

  throw std::logic_error("Axis should be:\nAXIS::ROWS or AXIS::COLUMNS");
}

Matrix Matrix::mean(AXIS axis) {
  std::size_t row_or_col = (axis == AXIS::ROW) ? rows_ : cols_;
  return sum(axis) / double(row_or_col);
}

Matrix Matrix::getCol(std::size_t col) const {

  if (col > cols_) {
    throw std::logic_error("Wrong column index.\n");
  }

  std::vector<double> resColumn;
  for (auto i = col; i < size(); i += cols_) {
    resColumn.push_back(data_[i]);
  }

  return Matrix(rows_, 1, resColumn);
}

Matrix Matrix::getRow(std::size_t row) const {
  if (row > rows_) {
    throw std::logic_error("Matrix getRow(std::size_t row) const:\nWrong column index.\n");
  }

  std::vector<double> resRow;
  for (auto i = 0; i < cols_; ++i) {
    resRow.push_back(get(row, i));
  }

  return Matrix(1, cols_, resRow);
}

void Matrix::printMatrix() const noexcept {
  std::cout << '{' << ' ';
  for (auto i = 0; i < size(); ++i) {
    if (i >= cols_ && i % cols_ == 0)
      std::cout << "  ";
    std::cout << data_[i] << ' ';
    if (i % cols_ == cols_ - 1 && rows_ != 1 && i != size() - 1)
      std::cout << '\n';
  }
  std::cout << '}' << std::endl;
}

Matrix Matrix::T() const {
  std::vector<double> transpose(size());

  for (std::size_t i = 0; i < rows_; ++i) {
    for (std::size_t j = 0; j < cols_; ++j) {
      transpose[j * rows_ + i] = data_[i * cols_ + j];
    }
  }

  return Matrix(cols_, rows_, transpose);
}

double Matrix::sum() const { return std::accumulate(data_.begin(), data_.end(), 0.0); }

double Matrix::L1() const {
  double norm = 0.0;
  for (auto d : data_) {
    norm += std::abs(d);
  }
  return norm;
}

double Matrix::L2() const {
  double norm = 0.0;
  for (auto d : data_) {
    norm += d * d;
  }
  return norm;
}

Matrix abs(const Matrix &mat) {
  auto data = mat.getData();
  for (auto &elem : data) {
    elem = std::abs(elem);
  }
  return Matrix(mat.rows(), mat.cols(), std::move(data));
}

} // namespace mlfs