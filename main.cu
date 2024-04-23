#include <cuda.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common.h"

// =================
// Helper Functions
// =================

// Command Line Option Processing
int find_arg_idx(int argc, char **argv, const char *option) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], option) == 0) {
      return i;
    }
  }
  return -1;
}

char *find_string_option(int argc, char **argv, const char *option,
                         char *default_value) {
  int iplace = find_arg_idx(argc, argv, option);

  if (iplace >= 0 && iplace < argc - 1) {
    return argv[iplace + 1];
  }

  return default_value;
}

// ==============
// Main Function
// ==============

int main(int argc, char **argv) {
  // Parse Args
  if (find_arg_idx(argc, argv, "-h") >= 0) {
    std::cout << "Options:" << std::endl;
    std::cout << "-h: see this help" << std::endl;
    std::cout << "-f: dataset csv file name" << std::endl;
    return 0;
  }

  char *dataset_file_name =
      find_string_option(argc, argv, "-f", "Admission_Predict.csv");

  std::ifstream dataset_file;
  dataset_file.open("datasets/" + dataset_file_name);
  if (!dataset_file.is_open()) {
    std::cerr << "Error opening dataset" << std::endl;
    return -1;
  }

  std::string line;
  // get past the header line
  std::getline(dataset_file, line);
  std::vector<std::vector<double>> csvData;
  while (std::getline(dataset_file, line)) {
    std::stringstream ss(line);
    std::vector<std::string> row;
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      row.push_back((double)cell);
    }
    csvData.push_back(row);
  }
  dataset_file.close();

  int N = csvData.size();
  // last dimension is the y_value
  int D = csvData[0].size() - 1;
  int split = ceil(0.8 * N);
  double *x_train = malloc(split * D * sizeof(double));
  double *x_test = malloc((N - split) * D * sizeof(double));
  double *y_train = malloc(split * sizeof(double));
  double *y_test = malloc((N - split) * sizeof(double));
  double *accuracy = malloc(sizeof(double));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      if (i < split) {
        x_train[i * D + j] = csvData[i][j];
      } else {
        x_test[i * D + j] = csvData[i][j];
      }
    }
    if (i < split) {
      y_train[i] = csvData[i][D];
    } else {
      y_test[i] = csvData[i][D];
    }
  }

  std::cout << "Size of x train: " << sizeof(x_train) / sizeof(x_train[0])
            << std::endl;

  // copy data to the gpu
  double *x_train_gpu;
  double *y_train_gpu;
  double *x_test_gpu;
  double *y_test_gpu;
  double *accuracy_gpu;
  cudaMalloc((void **)&x_train_gpu, split * D * sizeof(double));
  cudaMalloc((void **)&y_train_gpu, split * sizeof(double));
  cudaMalloc((void **)&x_test_gpu, (N - split) * D * sizeof(double));
  cudaMalloc((void **)&y_test_gpu, (N - split) * sizeof(double));
  cudaMalloc((void **)&accuracy_gpu, 1 * sizeof(double));

  cudaMemcpy(x_train_gpu, x_train, split * D * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(x_test_gpu, x_test, (N - split) * D * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(y_train_gpu, y_train, split * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(y_test_gpu, y_test, (N - split) * sizeof(double),
             cudaMemcpyHostToDevice);

  // track start time
  auto start_time = std::chrono::steady_clock::now();

  // train and predict

  cudaDeviceSynchronize();
  auto end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  double seconds = diff.count();

  cudaMemcpy(accuracy, accuracy_gpu, sizeof(double), cudaMemcpyDeviceToHost);

  // Finalize
  std::cout << "Training and Prediction Time = " << seconds << " seconds for "
            << num_parts << " particles." << std::endl;
  std::cout << "Accuracy = " << *accuracy << std::endl;
  cudaFree(x_train_gpu);
  cudaFree(x_test_gpu);
  cudaFree(y_train_gpu);
  cudaFree(y_test_gpu);
  free(x_train);
  free(x_test);
  free(y_train);
  free(y_test);
}
