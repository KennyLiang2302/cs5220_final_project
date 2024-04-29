#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sstream>

void write_to_csv(char *file_name, std::vector<double> data, int N, int D)
{
  std::ofstream file(file_name);
  if (!file)
  {
    std::cerr << "Unable to open file for writing." << std::endl;
  }

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < D; j++)
    {
      file << data[i * D + j];

      if (j != D - 1)
      {
        file << ",";
      }
    }
    file << std::endl;
  }

  file.close();
}