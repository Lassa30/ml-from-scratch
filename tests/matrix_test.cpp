#include <iostream>
#include <chrono>
#include "../include/utils/matrix.hpp"
#include <random>

int main() {
    // Define the size and bounds
    size_t vector_size = 10000;
    double lower_bound = 0.0, upper_bound = 1.0;

    // Set up the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    // Generate the random vector
    std::vector<double> random_vector1(vector_size);
    std::vector<double> random_vector2(vector_size);
    for (auto& value : random_vector1) value = dis(gen);
    for (auto& value : random_vector2) value = dis(gen);

    std::cout << random_vector1.size() << ' ' << random_vector2.size() << '\n';

    mlfs::Matrix mat1(100, 100, random_vector1);
    mlfs::Matrix mat2(100, 100, random_vector2);

    auto start = std::chrono::high_resolution_clock::now();
    mat1.matmul(mat2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
}