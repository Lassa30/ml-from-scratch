#ifndef UTILS_2024_09_06
#define UTILS_2024_09_06

#include <utils/matrix.hpp>

#include <algorithm>
#include <set>

#include <algorithm>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <vector>

namespace mlfs
{

    namespace utils
    {

        bool dataFromCsv(std::vector<double> &dataset, const std::string &filename, const char delim = ',')
        {

            auto datasetCp(dataset);

            std::fstream file(filename);
            if (!file.is_open())
            {
                std::cout << "In vectFromCsv():\n\t\tThe file wasn't opened\n";
                file.close();
                return false;
            }

            std::stringstream buf("");
            for (; !file.eof();)
            {
                char symb = file.get();
                if (symb == delim || symb == '\n')
                {
                    datasetCp.emplace_back(std::stod(buf.str()));
                    buf = std::stringstream("");
                }
                else
                {
                    buf << symb;
                }
            }
            dataset = std::move(datasetCp);

            file.clear();
            file.close();

            return true;
        }

        double accuracyScore(const Matrix &prediction, const Matrix &answer)
        {
            // prediction and answer should have equal shapes like: (1, n) or (n, 1)
            double right = 0;

            for (auto i = 0; i < prediction.rows_; i++)
            {
                for (auto j = 0; j < prediction.cols_; j++)
                {
                    right += (std::abs(prediction.get(i, j) - answer.get(i, j)) < EPSILON);
                }
            }

            return right / std::max(prediction.rows_, prediction.cols_);
        }
        void genIdx(std::set<int> &idxSet, const int left, const int right, const int cnt, int randomState = 42)
        {
            int setSize = 0;

            std::mt19937 gen(randomState);
            std::uniform_int_distribution<> dis(left, right);

            while (setSize < cnt)
            {
                int num = dis(gen);
                if (idxSet.insert(num).second)
                {
                    ++setSize;
                }
            }
        }

        void vectFromIdx(std::vector<double> &vect, const std::set<int> &setIdx, const mlfs::Matrix &designMatrixTrain)
        {
            for (auto i : setIdx)
            {
                std::vector<double> toPush(designMatrixTrain.get_row(i).get_data());
                vect.insert(vect.end(), toPush.begin(), toPush.end());
            }
        }
    }
} // mlfs::utils

#endif // UTILS_2024_09_06