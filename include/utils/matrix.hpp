#ifndef MATRIX_HPP_d08_m26_y24
#define MATRIX_HPP_d08_m26_y24

#include <cstdint>
#include <iostream>
#include <vector>

namespace mlfs { // MLFS - MlFromScratch

// let epsilon be 2 * 2^-52 = 2^-51 ~ 4.4408921e-16
// source: //
// https://stackoverflow.com/questions/13698927/compare-double-to-zero-using-epsilon

#define EPSILON 4.4408921e-16

class Matrix {
private:
  // data storage
  std::vector<double> data_{};

  // number of rows | ax0
  std::size_t rows_;

  // number of columns | ax1
  std::size_t cols_;

  void swap(Matrix &rhs) noexcept;

  std::string shape_err_msg(const std::string &method, const Matrix &rhs) const;

public:
  enum AXIS { ROW, COLUMN };

  // default ctor: do nothing - all members are initialized by default
  Matrix() : data_{}, rows_{0}, cols_{0} {};

  // fill-constructor: fill the Matrix with the value
  Matrix(const std::size_t &rows, const std::size_t &columns);

  // from-data-ctor
  Matrix(const std::size_t &rows, const std::size_t &columns, const std::vector<double> &rhs);

  // flat-std::vector-constructor
  Matrix(const std::vector<double> &rhs, AXIS axis);

  Matrix(const Matrix &rhs) = default;

  Matrix &operator=(const Matrix &rhs);

  // Move semantics
  Matrix(Matrix &&rhs) noexcept;

  Matrix &operator=(Matrix &&rhs) noexcept;

  inline std::pair<std::size_t, std::size_t> shape() const { return {rows_, cols_}; }

  void reshape(const std::size_t &cols, const std::size_t &rows);

  // Algebra
  Matrix &operator+=(const Matrix &rhs);

  Matrix &operator+=(const double &rhs);

  Matrix &operator-=(const Matrix &rhs);

  Matrix &operator*=(const double &rhs);

  Matrix &operator/=(const double &rhs);

  Matrix operator+(const double &rhs) const;
  Matrix operator+(const Matrix &rhs) const;

  Matrix operator-(const Matrix &rhs) const;
  Matrix operator-() const;

  Matrix operator*(const double &rhs) const;

  Matrix operator*(const Matrix &rhs) const;

  Matrix operator/(const double &rhs) const;

  Matrix matmul(const Matrix &rhs) const;

  Matrix sum(AXIS axis);

  Matrix mean(AXIS axis);

  Matrix T() const;

  double sum() const;

  double L1() const;

  double L2() const;

  // Getters

  Matrix getCol(std::size_t col) const;

  Matrix getRow(std::size_t row) const;

  inline double get(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

  inline std::size_t rows() const { return rows_; }

  inline std::size_t cols() const { return cols_; }

  inline std::size_t size() const { return cols_ * rows_; }

  inline std::vector<double> getData() const { return data_; }

  void printMatrix() const noexcept;

};

Matrix abs(const Matrix &mat);

} // namespace mlfs
#endif // MATRIX_HPP_d08_m26_y24