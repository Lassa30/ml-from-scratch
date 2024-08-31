#include <utils/matrix.hpp>

#include <algorithm>
#include <set>

#include <fstream>
#include <sstream>

namespace mlfs {

namespace utils {

bool dataFromCsv(std::vector<double> & dataset, std::string filename) {

    auto datasetCp(dataset);

    std::fstream file(filename);
    if (!file.is_open()) {
        std::cout << "In vectFromCsv():\n\t\tThe file wasn't opened\n";
        file.close();
        return false;
    }

    std::stringstream buf("");
    for (;!file.eof();) {
        char symb = file.get();
        if (symb == ',' || symb == '\n') {
            datasetCp.emplace_back(std::stod(buf.str()));
            buf = std::stringstream("");
        }else {
            buf << symb;
        }
    }
    dataset = std::move(datasetCp);

    file.clear();
    file.close();

    return true;
}

double accuracyScore(const mlfs::Matrix & prediction, const mlfs::Matrix & answer) {
    // prediction and answer should have equal shapes like: (1, n) or (n, 1)
    double right = 0;

    for (auto i = 0; i < prediction.rows_; i++) {
        for (auto j = 0; j < prediction.cols_; j++) {
            right += (std::abs(prediction.get(i, j) - answer.get(i, j)) < EPSILON);
        }
    }

    return right / std::max(prediction.rows_, prediction.cols_);
}

}} // mlfs::utils