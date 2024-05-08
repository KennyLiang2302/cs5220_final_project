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
#include "common.h"

std::string model_type;

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

double get_seconds(std::chrono::time_point<std::chrono::steady_clock> start, std::chrono::time_point<std::chrono::steady_clock> end)
{
  std::chrono::duration<double> diff = end - start;
  return diff.count();
}

void run(double *x_train, double *y_train, double *x_test, double *y_test, int D, int total_size, int train_size, int test_size)
{
  std::vector<double> x_train_vec(x_train, x_train + D * train_size);
  std::vector<double> y_train_vec(y_train, y_train + train_size);
  std::vector<double> x_test_vec(x_test, x_test + D * test_size);
  std::vector<double> y_test_vec(y_test, y_test + test_size);

  init(D, train_size, test_size, x_train_vec, y_train_vec, x_test_vec, y_test_vec);

  // track start time
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> forest_training_end_time;
  std::chrono::time_point<std::chrono::steady_clock> forest_inference_end_time;
  std::chrono::time_point<std::chrono::steady_clock> tree_training_end_time;
  std::chrono::time_point<std::chrono::steady_clock> tree_inference1_end_time;
  std::chrono::time_point<std::chrono::steady_clock> tree_inference2_end_time;
  std::chrono::time_point<std::chrono::steady_clock> tree_iter_training_end_time;
  std::chrono::time_point<std::chrono::steady_clock> tree_iter_inference1_end_time;

  double forest_mse;
  double tree_mse;
  double tree_class_error;
  double tree_iter_mse;
  double tree_iter_class_error;

  if (model_type == "rf" || model_type == "all")
  {
    forest_t *forest = build_forest(DEPTH, NUM_TREES);
    forest_training_end_time = std::chrono::steady_clock::now();

    forest_mse = eval_forest_mse(forest);
    forest_inference_end_time = std::chrono::steady_clock::now();
  }

  if (model_type == "dtr" || model_type == "all")
  {
    tree_node_t *tree = build_cart(DEPTH);
    tree_training_end_time = std::chrono::steady_clock::now();

    tree_mse = eval_mse(tree);
    tree_inference1_end_time = std::chrono::steady_clock::now();

    tree_class_error = eval_classification(tree);
    tree_inference2_end_time = std::chrono::steady_clock::now();
  }

  if (model_type == "dti" || model_type == "all")
  {
    tree_node_t *tree = build_cart_iterative(DEPTH);
    tree_iter_training_end_time = std::chrono::steady_clock::now();

    tree_iter_mse = eval_mse(tree);
    tree_iter_inference1_end_time = std::chrono::steady_clock::now();

    tree_iter_class_error = eval_classification(tree);
  }

  if (model_type == "rf" || model_type == "all")
  {
    std::cout << "Forest training time = " << get_seconds(start_time, forest_training_end_time) << std::endl;
    std::cout << "Forest inference time = " << get_seconds(forest_training_end_time, forest_inference_end_time) << std::endl;
    std::cout << "Forest MSE = " << forest_mse << std::endl;
    std::cout << "=========================================================" << std::endl;
  }

  if (model_type == "dtr" || model_type == "all")
  {
    auto dtr_start_time = model_type == "all" ? forest_inference_end_time : start_time;
    std::cout << "Decision Tree Recursive training time = " << get_seconds(dtr_start_time, tree_training_end_time) << std::endl;
    std::cout << "Decision Tree Recursive inference time = " << get_seconds(tree_training_end_time, tree_inference1_end_time) << std::endl;
    std::cout << "Decision Tree Recursive MSE = " << tree_mse << std::endl;
    std::cout << "Decision Tree Recursive Classification Error = " << tree_class_error << std::endl;
    std::cout << "=========================================================" << std::endl;
  }

  if (model_type == "dti" || model_type == "all")
  {
    auto dti_start_time = model_type == "all" ? tree_inference2_end_time : start_time;
    std::cout << "Decision Tree Iterative training time = " << get_seconds(dti_start_time, tree_iter_training_end_time) << std::endl;
    std::cout << "Decision Tree Iterative inference time = " << get_seconds(tree_iter_training_end_time, tree_iter_inference1_end_time) << std::endl;
    std::cout << "Decision Tree Iterative MSE = " << tree_iter_mse << std::endl;
    std::cout << "Decision Tree Iterative Classification Error = " << tree_iter_class_error << std::endl;
    std::cout << "=========================================================" << std::endl;
  }
}

int main(int argc, char **argv)
{
  // Parse Args
  if (find_arg_idx(argc, argv, "-h") >= 0)
  {
    std::cout << "Options:" << std::endl;
    std::cout << "-h: see this help" << std::endl;
    std::cout << "-f: dataset csv file name" << std::endl;
    std::cout << "-m: model type (rf or dtr or dti or all)" << std::endl;
    return 0;
  }

  char *dataset_file_name =
      find_string_option(argc, argv, "-f", "../datasets/Admission_Predict.csv");

  model_type = find_string_option(argc, argv, "-m", "all");

  if (model_type != "dtr" && model_type != "rf" && model_type != "dti" && model_type != "all")
  {
    std::cerr << "Model type must be either rf, dti, dtr, or all" << std::endl;
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
  std::cout << "=========================================================" << std::endl;
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
