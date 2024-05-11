#ifndef __COMMON_H__
#define __COMMON_H__

#include <cfloat>
#include <vector>

#define DEPTH 10
#define NUM_THREADS 1
#define NUM_STREAMS 1
#define THRESHOLD 100 * DBL_EPSILON
#define WRITE_TO_CSV false
#define TRAIN_TEST_SPLIT 0.8
#define NUM_TREES 1000
#define SUBSAMPLE_RATE 0.8

typedef struct tree_node_t
{
  tree_node_t *left;
  tree_node_t *right;
  tree_node_t *parent;
  int cut_feature;
  double cut_value;
  double prediction;
} tree_node_t;

using forest_t = std::vector<tree_node_t *>;

typedef struct split_output_t
{
  int cut_feature;
  double cut_value;
  double loss;
} split_output_t;

void init(int D, int train_size, int test_size, std::vector<double> &x_train, std::vector<double> &y_train, std::vector<double> &x_test, std::vector<double> &y_test);

tree_node_t *build_cart(int depth);

tree_node_t *build_cart_iterative(int depth);

double eval_mse(tree_node_t *tree);

double eval_classification(tree_node_t *tree);

forest_t *build_forest(int depth, int num_trees);

double eval_forest_mse(forest_t *forest);
#endif