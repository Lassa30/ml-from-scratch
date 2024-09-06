#include <utils/matrix.hpp>

#include <iostream>
#include <chrono>
#include <random>

typedef std::chrono::high_resolution_clock measure;
typedef std::chrono::duration<double, std::milli> dur_ms;

int main()
{
    // Create vectors filled with random numbers
    size_t vector_size = 1'000'000;
    double lower_bound = 0.0, upper_bound = 1.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    std::vector<double> random_vector1(vector_size);
    std::vector<double> random_vector2(vector_size);

    for (auto &value : random_vector1)
        value = dis(gen);
    for (auto &value : random_vector2)
        value = dis(gen);

    // Define matrices...
    mlfs::Matrix mat1(1'000, 1'000, random_vector1);
    mlfs::Matrix mat2(1'000, 1'000, random_vector2);

    // Run matmul
    auto start = measure::now();
    mat1.matmul(mat2);
    auto end = std::chrono::high_resolution_clock::now();
    dur_ms duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    std::cout << "Transpose testing.\nBefore T():\n";
    auto transposeTest = mlfs::Matrix(3, 2, {1, 2, 3, 4, 5, 6});
    transposeTest.print_matrix();
    std::cout << "After T():\n";
    transposeTest.T().print_matrix();

    transposeTest = mlfs::Matrix(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    transposeTest.T().print_matrix();

    // Benchmarks:
    // square matrices: 1'000 x 1'000
    // First: naive algorithm:
    // It took 21.258 seconds...
    // Second: naive algorithm + compiler flags:
    // ~ 900ms
    // Third: Strassen's algorithm
    // maybe later...
}