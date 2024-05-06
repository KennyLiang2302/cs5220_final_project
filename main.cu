#include <cuda.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include "util.h"
#include "common.h"

bool computeMSE = true;
bool trainRF = false;

// Command Line Option Processing
int find_arg_idx(int argc, char **argv, const char *option)
{
  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], option) == 0)
    {
      return i;
    }
  }
  return -1;
}

char *find_string_option(int argc, char **argv, const char *option,
                         char *default_value)
{
  int iplace = find_arg_idx(argc, argv, option);

  if (iplace >= 0 && iplace < argc - 1)
  {
    return argv[iplace + 1];
  }

  return default_value;
}

void run(double *x_train, double *y_train, double *x_test, double *y_test, int D, int total_size, int train_size, int test_size)
{
  std::vector<double> x_train_vec(x_train, x_train + D * train_size);
  std::vector<double> y_train_vec(y_train, y_train + train_size);
  std::vector<double> x_test_vec(x_test, x_test + D * test_size);
  std::vector<double> y_test_vec(y_test, y_test + test_size);

  if (WRITE_TO_CSV)
  {
    // create a csv with xtrain and ytrain data
    write_data_to_csv("../datasets/x_train.csv", x_train_vec, train_size, D);
    write_data_to_csv("../datasets/y_train.csv", y_train_vec, train_size, 1);
  }

  // track start time
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> training_end_time;
  std::chrono::time_point<std::chrono::steady_clock> inference_end_time;
  double error;

  if (trainRF)
  {
    forest_t *forest = build_forest(D, train_size, x_train_vec, y_train_vec, DEPTH, NUM_TREES);
    training_end_time = std::chrono::steady_clock::now();
    error = eval_forest_mse(D, test_size, x_test_vec, y_test_vec, forest);
    inference_end_time = std::chrono::steady_clock::now();
  }
  else
  {
    tree_node_t *tree_node = build_cart(D, train_size, x_train_vec, y_train_vec, DEPTH);
    training_end_time = std::chrono::steady_clock::now();

    if (computeMSE)
    {
      error = eval_mse(D, test_size, x_test_vec, y_test_vec, tree_node);
    }
    else
    {
      error = eval_classification(D, train_size, x_train_vec, y_train_vec, tree_node);
    }

    inference_end_time = std::chrono::steady_clock::now();
  }

  std::chrono::duration<double> training_diff = training_end_time - start_time;
  double training_seconds = training_diff.count();

  std::chrono::duration<double> inference_diff = inference_end_time - training_end_time;
  double inference_seconds = inference_diff.count();

  std::chrono::duration<double> total_diff = inference_end_time - start_time;
  double total_seconds = total_diff.count();

  std::cout << "Training and Inference Time = " << total_seconds << std::endl;
  std::cout << "Training Time = " << training_seconds << std::endl;
  std::cout << "Inference Time = " << inference_seconds << std::endl;
  std::cout << "Error = " << error << std::endl;
}

int main(int argc, char **argv)
{
  // Parse Args
  if (find_arg_idx(argc, argv, "-h") >= 0)
  {
    std::cout << "Options:" << std::endl;
    std::cout << "-h: see this help" << std::endl;
    std::cout << "-f: dataset csv file name" << std::endl;
    std::cout << "-e: evaluation error metric (mse or class)" << std::endl;
    std::cout << "-m: model type (rf or dt)" << std::endl;
    return 0;
  }

  char *dataset_file_name =
      find_string_option(argc, argv, "-f", "../datasets/Admission_Predict.csv");

  std::string eval_type =
      find_string_option(argc, argv, "-e", "mse");

  if (eval_type == "mse")
  {
    computeMSE = true;
  }
  else if (eval_type == "class")
  {
    computeMSE = false;
  }
  else
  {
    std::cerr << "Error metric must be either mse or class" << std::endl;
    return -1;
  }

  std::string model_type =
      find_string_option(argc, argv, "-m", "dt");

  if (model_type == "dt")
  {
    trainRF = false;
  }
  else if (model_type == "rf")
  {
    trainRF = true;
  }
  else
  {
    std::cerr << "Model type must be either rf or dt" << std::endl;
    return -1;
  }

  std::ifstream dataset_file;
  dataset_file.open(dataset_file_name);
  if (!dataset_file.is_open())
  {
    std::cerr << "Error opening dataset" << std::endl;
    return -1;
  }

  std::string line;
  // get past the header line
  std::getline(dataset_file, line);
  std::vector<std::vector<double>> csvData;
  while (std::getline(dataset_file, line))
  {
    std::stringstream ss(line);
    std::vector<double> row;
    std::string cell;
    while (std::getline(ss, cell, ','))
    {
      row.push_back(std::stod(cell));
    }
    csvData.push_back(row);
  }
  dataset_file.close();

  int N = csvData.size();
  // last dimension is the y_value
  int D = csvData[0].size() - 1;
  std::cout << "Dataset size is " << N << std::endl;
  std::cout << "Dataset dimension is " << D << std::endl;
  int train_size = ceil(TRAIN_TEST_SPLIT * N);
  int test_size = N - train_size;

  double *x_train = (double *)malloc(train_size * D * sizeof(double));
  double *x_test = (double *)malloc(test_size * D * sizeof(double));
  double *y_train = (double *)malloc(train_size * sizeof(double));
  double *y_test = (double *)malloc(test_size * sizeof(double));
  double *accuracy = (double *)malloc(sizeof(double));

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < D; ++j)
    {
      if (i < train_size)
      {
        x_train[i * D + j] = csvData[i][j];
      }
      else
      {
        x_test[(i - train_size) * D + j] = csvData[i][j];
      }
    }
    if (i < train_size)
    {
      y_train[i] = csvData[i][D];
    }
    else
    {
      y_test[i - train_size] = csvData[i][D];
    }
  }

  run(x_train, y_train, x_test, y_test, D, N, train_size, test_size);

  free(x_train);
  free(x_test);
  free(y_train);
  free(y_test);
  free(accuracy);
}
