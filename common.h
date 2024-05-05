#ifndef __COMMON_H__
#define __COMMON_H__

#include <cfloat>
#include <vector>

#define DEPTH 10
#define NUM_THREADS 2
#define THRESHOLD 100 * DBL_EPSILON
#define WRITE_TO_CSV false

typedef struct tree_node_t
{
  tree_node_t *left;
  tree_node_t *right;
  tree_node_t *parent;
  int cut_feature;
  double cut_value;
  double prediction;
} tree_node_t;

typedef struct forest_t
{
  std::vector<tree_node_t *> trees;
} forest_t;

typedef struct split_output_t
{
  int cut_feature;
  double cut_value;
  double loss;
} split_output_t;

// function for tree construction. xtrain and ytrain are both in cpu memory
tree_node_t *build_cart(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train, int depth);

// function for making predictions on a dataset and computing accuracy. xtest
// and ytest are both in cpu memory
double eval_mse(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree);

double eval_classification(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree);

forest_t *build_forest(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train, int depth, int num_trees);

double eval_forest(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, forest_t *forest);
#endif