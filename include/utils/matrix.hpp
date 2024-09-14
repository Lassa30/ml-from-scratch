#ifndef MATRIX_HPP_d08_m26_y24
#define MATRIX_HPP_d08_m26_y24

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mlfs { // MLFS - MlFromScratch

// let epsilon be 2 * 2^-52 = 2^-51 ~ 4.4408921e-16
// source: //
// https://stackoverflow.com/questions/13698927/compare-double-to-zero-using-epsilon
#define EPSILON 4.4408921e-16

bool close_to(const double &lhs, const double &rhs) { return std::abs(lhs - rhs) < EPSILON; }

class Matrix {
private:
  // data storage
  std::vector<double> data_{};

  // number of rows | ax0
  std::size_t rows_ = 0;

  // number of columns | ax1
  std::size_t cols_ = 0;

  void swap(Matrix &rhs) noexcept {
    std::swap(rows_, rhs.rows_);
    std::swap(cols_, rhs.cols_);
    std::swap(data_, rhs.data_);
  }

  std::string shape_err_msg(const std::string &method, const Matrix &rhs) const {
    std::stringstream err_msg;

    err_msg << method << '\n';
    err_msg << "The shapes don't fit\n";

    err_msg << "lhs: (" << shape().first << " ," << shape().second << ")\n";

    err_msg << "rhs: (" << rhs.shape().first << " ," << rhs.shape().second << ")\n";

    return err_msg.str();
  }

public:
  enum AXIS { ROW, COLUMN };

  // default ctor: do nothing - all members are initialized by default
  Matrix(){};

  // fill-constructor: fill the Matrix with the value
  Matrix(const std::size_t &rows, const std::size_t &columns) : cols_{columns}, rows_{rows} {
    data_ = std::vector<double>(size(), 0.0);
  }

  // from-data-ctor
  Matrix(const std::size_t &rows, const std::size_t &columns, const std::vector<double> &rhs)
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

  // flat-std::vector-constructor
  Matrix(const std::vector<double> &rhs, AXIS axis) {
    if (rhs.empty()) {
      throw std::logic_error("Couldn't construct a Matrix from an empty std::vector<double>\n");
    }
    if (axis != AXIS::ROW && axis != AXIS::COLUMN) {
      throw std::logic_error("Invalid axis. Use AXIS::ROW or AXIS::COLUMN.");
    }

    rows_ = (axis == AXIS::ROW) ? 1 : rhs.size();
    cols_ = (axis == AXIS::ROW) ? rhs.size() : 1;

    data_ = std::vector<double>(rhs);
  };

  Matrix(const Matrix &rhs) : rows_{rhs.rows_}, cols_{rhs.cols_}, data_{rhs.data_} {};

  Matrix &operator=(const Matrix &rhs) {
    // copy 'n' swap idiom
    if (this != &rhs) {
      Matrix cp(rhs);
      swap(cp);
    }
    return *this;
  };

  // Move semantics
  Matrix(Matrix &&rhs) noexcept : rows_{rhs.rows_}, cols_{rhs.cols_}, data_{std::move(rhs.data_)} {
    rhs.rows_ = 0;
    rhs.cols_ = 0;
  }

  Matrix &operator=(Matrix &&rhs) noexcept {
    if (this != &rhs) {
      swap(rhs);
    }
    return *this;
  }

  inline std::pair<std::size_t, std::size_t> shape() const { return {rows_, cols_}; }

  inline void reshape(const std::size_t &cols, const std::size_t &rows) {
    if (cols * rows == 0) {
      throw std::logic_error("LogicError: Reshaping to zero sized Matrix is impossible.\n");
    }

    if (cols * rows == size()) {
      cols_ = cols;
      rows_ = rows;
    }
  }

  Matrix &operator+=(const Matrix &rhs) {
    if (!(rows_ == rhs.rows_ && cols_ == rhs.cols_)) {
      throw std::logic_error(shape_err_msg("Matrix & operator+=(const Matrix & rhs)", rhs));
    }

    for (auto i = 0; i < size(); ++i) {
      data_[i] += rhs.data_[i];
    }

    return *this;
  }

  Matrix &operator+=(const double &rhs) {
    for (auto i = 0; i < size(); ++i) {
      data_[i] += rhs;
    }

    return *this;
  }

  Matrix &operator-=(const Matrix &rhs) {
    if (!(rows_ == rhs.rows_ && cols_ == rhs.cols_)) {
      throw std::logic_error(shape_err_msg("Matrix & operator-=(const Matrix &rhs)", rhs));
    }

    for (auto i = 0; i < size(); ++i) {
      data_[i] -= rhs.data_[i];
    }

    return *this;
  }

  Matrix &operator*=(const double &rhs) {
    for (auto &elem : data_)
      elem *= rhs;

    return *this;
  }

  Matrix &operator/=(const double &rhs) {
    if (close_to(rhs, 0.0))
      throw std::runtime_error("Zero division. Matrix / 0.0 is not defined");

    for (auto &elem : data_)
      elem /= rhs;

    return *this;
  }

  Matrix operator+(const double &rhs) const {
    Matrix temp(*this);
    temp += rhs;
    return temp;
  }

  Matrix operator+(const Matrix &rhs) const {
    Matrix temp(*this);
    temp += rhs;
    return temp;
  }

  Matrix operator-(const Matrix &rhs) const {
    Matrix temp(*this);
    temp -= rhs;
    return temp;
  }

  Matrix operator-() const {
    auto temp(*this);
    std::size_t i = 0;

    for (auto &elem : temp.data_)
      elem = -elem;

    return temp;
  }

  Matrix operator*(const double &rhs) const {
    Matrix temp(*this);
    temp *= rhs;
    return temp;
  }

  Matrix operator*(const Matrix &rhs) const {
    if (!(cols_ == rhs.cols_ && rows_ == rhs.rows_)) {
      throw std::logic_error("Matrices should have equal shapes.\n");
    }

    std::vector<double> cp(size(), 1.0);
    for (auto i = 0; i < size(); i++) {
      cp[i] *= data_[i] * rhs.data_[i];
    }

    return Matrix(rows_, cols_, cp);
  }

  Matrix operator/(const double &rhs) const {
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

  Matrix matmul(const Matrix &rhs) const {

    if (cols_ != rhs.rows_) {
      throw std::logic_error(shape_err_msg("Matrix matmul(const Matrix & rhs)", rhs));
    }

    Matrix product(rows_, rhs.cols_, std::vector<double>(rows_ * rhs.cols_, 0.0));

    // Naive algorithm is better for small matrices.
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

  Matrix sum(AXIS axis) {
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

  Matrix mean(AXIS axis) {
    std::size_t row_or_col = (axis == AXIS::ROW) ? rows_ : cols_;
    return sum(axis) / double(row_or_col);
  }

  double get(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

  Matrix getCol(std::size_t col) const {

    if (col > cols_) {
      throw std::logic_error("Wrong column index.\n");
    }

    std::vector<double> resColumn;
    for (auto i = col; i < size(); i += cols_) {
      resColumn.push_back(data_[i]);
    }

    return Matrix(rows_, 1, resColumn);
  }

  Matrix getRow(std::size_t row) const {
    if (row > rows_) {
      throw std::logic_error("Matrix getRow(std::size_t row) const:\nWrong column index.\n");
    }

    std::vector<double> resRow;
    for (auto i = 0; i < cols_; ++i) {
      resRow.push_back(get(row, i));
    }

    return Matrix(1, cols_, resRow);
  }

  inline std::size_t size() const { return cols_ * rows_; }

  inline void printMatrix() const noexcept {
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

  std::vector<double> getData() const { return data_; }

  Matrix T() const {
    std::vector<double> transpose(size());

    for (std::size_t i = 0; i < rows_; ++i) {
      for (std::size_t j = 0; j < cols_; ++j) {
        transpose[j * rows_ + i] = data_[i * cols_ + j];
      }
    }

    return Matrix(cols_, rows_, transpose);
  }

  double sum() const { return std::accumulate(data_.begin(), data_.end(), 0.0); }

  double L1() const {
    double norm = 0.0;
    for (auto d : data_) {
      norm += std::abs(d);
    }
    return norm;
  }

  double L2() const {
    double norm = 0.0;
    for (auto d : data_) {
      norm += d * d;
    }
    return norm;
  }

  inline std::size_t rows() const { return rows_; }
  inline std::size_t cols() const { return cols_; }

}; // namespace mlfs - MlFromScratch
} // namespace mlfs
#endif // MATRIX_HPP_d08_m26_y24