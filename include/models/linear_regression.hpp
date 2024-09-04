#include <utils/matrix.hpp>

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

namespace mlfs
{
    // Linear Regression with no normalization.
    class LinearRegression
    {
    public:
        void virtual train(const Matrix &features, const Matrix &target)
        {

        }

        void virtual predict(const Matrix &features) 
        {

        }
    private:
        Matrix solution_{};
        
    };
}