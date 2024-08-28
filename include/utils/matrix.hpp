#ifndef MATRIX_HPP_d08_m26_y24
#define MATRIX_HPP_d08_m26_y24

#include <algorithm>
#include <sstream>
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>



namespace mlfs { // MLFS - MlFromScratch

using std::vector, std::size_t;

typedef std::shared_ptr<std::vector<double>> sPtr;

// let epsilon be 2 * 2^-52 = 2^-51 ~ 4.4408921e-16
// source: // https://stackoverflow.com/questions/13698927/compare-double-to-zero-using-epsilon

#define EPSILON 4.4408921e-16 

bool close_to(const double & lhs, const double & rhs) {
    return std::abs(lhs - rhs) < EPSILON;
}

class Matrix {
private:
    // data storage
    sPtr data_ = nullptr;

    // number of rows | ax0
    size_t rows_ = 0;

    // number of columns | ax1
    size_t cols_ = 0;

    void swap(Matrix & rhs) noexcept {
        std::swap(rows_, rhs.rows_);
        std::swap(cols_, rhs.cols_);
        std::swap(data_, rhs.data_);
    }

    std::string shape_err_msg(const std::string & method, const Matrix & rhs) {
        std::stringstream err_msg;

        err_msg << method << '\n';
        err_msg << "The shapes don't fit\n";

        err_msg << "lhs: (" << shape().first << " ,"
                            << shape().second << ")\n";

        err_msg << "rhs: (" << rhs.shape().first << " ," 
                            << rhs.shape().second << ")\n";

        return err_msg.str();
    }

public:

    enum AXIS {
        ROW,
        COLUMN
    };

    // default ctor: do nothing - all members are initialized by default
    Matrix() {};

    // fill-constructor: fill the Matrix with the value
    Matrix(const size_t & rows, const size_t & columns)
    : cols_{columns}
    , rows_{rows}
    {
        data_ = std::make_shared<std::vector<double>>(cols_ * rows_);
    }

    // from-data-ctor
    Matrix(
        const size_t & rows, 
        const size_t & columns, 
        const vector<double> & rhs) 

        : rows_{rows}
        , cols_{columns}
    {

        if (rows_ * cols_ != rhs.size()) {
            throw std::logic_error(
                "Unable to construct a Matrix with these parameters: rows_ * cols_ != rhs.size()\n"
            );
        }

        data_ = std::make_shared<vector<double>>(rhs);
    }

    // flat-vector-constructor
    Matrix(const vector<double> & rhs, AXIS axis) {
        
        /*
        const vector<double> & rhs: 
            shouldn't be an empty vector

        int axis:
            AXIS::ROW -> making a (1, n) matrix
            AXIS::COLUMN -> a (n, 1) matrix
        */

        if (rhs.empty()) {
            throw std::logic_error(
                "Couldn't construct a Matrix from an empty std::vector<double>\n");
        }
        if (axis == AXIS::ROW) {
            rows_ = 1;
            cols_ = rhs.size();
        } else if (axis == AXIS::COLUMN) {
            rows_ = rhs.size();
            cols_ = 1;
        } else {
            throw std::logic_error("Invalid axis. Use AXIS::ROW or AXIS::COLUMN.");
        }

        data_ = std::make_shared<std::vector<double>>(rhs);
    };

    // Sharing the data_, doesn't perform a deep copying
    Matrix(const Matrix & rhs) 
    : rows_{rhs.rows_}
    , cols_{rhs.cols_}
    , data_{rhs.data_}
    {
    };

    Matrix & operator=(const Matrix & rhs) {
        // copy 'n' swap
        if (this != &rhs) {
            Matrix cp(rhs);

            swap(cp);
        }
        return *this;
    };

    // Deep Copy member function for special cases
    Matrix copy() {
        Matrix deepcopy(rows_, cols_);
        
        std::copy(
            data_->begin(), 
            data_->end(), 
            deepcopy.data_->begin()
        );

        return deepcopy;
    }

    // Move semantics
    Matrix(Matrix && rhs) noexcept 
        : rows_{rhs.rows_}
        , cols_{rhs.cols_}
        , data_{std::move(rhs.data_)}
    {
        rhs.rows_ = 0;
        rhs.cols_ = 0;
    }

    Matrix & operator=(Matrix && rhs) noexcept {
        if (this != &rhs) {
            swap(rhs);
        }
        return *this;
    }

    inline std::pair<size_t, size_t> shape() const { 
        return {cols_, rows_};
    }

    inline void reshape(const size_t & lhs, const size_t & rhs) {
        if (lhs * rhs == 0) {
            throw std::logic_error("LogicError: Reshaping to zero sized Matrix is impossible.\n");
        }

        if (lhs * rhs == cols_ * rows_) {
            cols_ = lhs;
            rows_ = rhs;
        }
    }
    

    /*begin: Arithmetics*/

    Matrix & operator+=(const Matrix & rhs) {
        if (!(rows_ == rhs.rows_ && cols_ == rhs.cols_)) {
            throw std::logic_error(
                shape_err_msg("Matrix & operator+=(const Matrix & rhs)", rhs)
            );
        }

        for (auto i = 0; i < rows_ * cols_; ++i) {
            (*data_)[i] += (*rhs.data_)[i];
        }

        return *this;
    }

    Matrix & operator+=(const double & rhs) {
        std::for_each(data_->begin(), 
                      data_->end(), 
                      [&rhs](double & d) {d += rhs;}
        );

        return *this;
    }

    Matrix & operator*=(const double & rhs) {
        std::for_each(data_->begin(), 
                      data_->end(), 
                      [&rhs](double & d) {d *= rhs;}
        );

        return *this;
    }

    Matrix & operator/=(const double & rhs) {
        if (close_to(rhs, 0.0))
            throw std::runtime_error(
                "Zero division. Matrix / 0.0 is not defined"
            );

        std::for_each(data_->begin(), 
                      data_->end(), 
                      [&rhs](double & d) {d /= rhs;}
        );

        return *this;
    }

    Matrix operator+(const double & rhs) {
        Matrix temp(rows_, cols_);
        temp += rhs;
        return temp;
    }

    Matrix operator+(const Matrix & rhs) {
        Matrix temp = copy();
        temp += rhs;
        return temp;
    }

    Matrix operator-() {
        vector<double> temp(rows_ * cols_);
        size_t i = 0;

        std::for_each(temp.begin(), 
                      temp.end(), 
                      [this, &i](double& d){d = -(*data_)[i], i++;}
        );

        return Matrix(rows_, cols_, temp);
    }

    Matrix operator*(const double & rhs) {
        Matrix temp = copy();
        temp *= rhs;
        return temp;
    }

    Matrix operator/(const double & rhs) {
        if (close_to(rhs, 0.0)) {
            throw std::runtime_error(
                "Zero division.\n" \
                "Method: Matrix operator/(const double & rhs)\n" \
                "rhs=" + std::to_string(rhs) + '\n'
            );
        }

        return *this * (1 / rhs);
    }

    // Naive algorithm works very slow:
    // 
    // Strassen algorithm is going to be used.

    Matrix matmul(const Matrix & rhs) {

        if (cols_ != rhs.rows_) {
            throw std::logic_error(
                shape_err_msg("Matrix matmul(const Matrix & rhs)", rhs)
            );
        }

        Matrix product(rows_, rhs.cols_);

        // Naive algorithm is better for small matrices. 
        // source: wikipedia

        for (auto i = 0; i < rows_; ++i) {
        
            for (auto j = 0; j < rhs.cols_; ++j) {

                for (auto k = 0; k < cols_; ++k) {
                    product.get(i, j) += get(i, k) * rhs.get(k, j);
                }

            }
        }

        return product;
    }
    
    Matrix sum(AXIS axis) {
        if (axis == AXIS::ROW || axis == AXIS::COLUMN) {
            size_t row_or_col = (axis == AXIS::ROW) ? rows_ : cols_;

            vector<double> result(row_or_col);

            // REFACTOR
            for (auto i = 0; i < rows_ * cols_; ++i) {
                result[i / row_or_col] += (*data_)[i];
            }

            return Matrix(result, axis);
        }else {
            throw std::logic_error("Axis should be:\nAXIS::ROWS or AXIS::COLUMNS");
        }
    }

    Matrix mean(AXIS axis) {
        size_t row_or_col = (axis == AXIS::ROW) ? rows_ : cols_;
        return sum(axis) / double(row_or_col);
    }

    /*end: Arithmetics*/

    double & get(size_t i, size_t j) const {
        return (*data_)[i * cols_ + j];
    }

    Matrix get_col(size_t col) {

        if (col > cols_) {
            throw std::logic_error("Wrong column index.\n");
        }

        vector<double> resColumn(rows_);
        for (auto i = col; i < rows_ * cols_; i += cols_) {
            resColumn[i / cols_] = (*data_)[i];
        }

        return Matrix(1, rows_, resColumn);
    }

    inline size_t size() const {
        return cols_ * rows_;
    }

};
} // namespace mlfs - MlFromScratch

#endif // MATRIX_HPP_d08_m26_y24