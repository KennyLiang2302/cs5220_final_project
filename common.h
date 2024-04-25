#ifndef __COMMON_H__
#define __COMMON_H__

#define DEPTH 50
#define NUM_BLOCKS 1
#define NUM_THREADS 1
#define THRESHOLD 1e-6

typedef struct tree_node_t
{
  tree_node_t *left;
  tree_node_t *right;
  tree_node_t *parent;
  int cut_feature;
  double cut_value;
  double prediction;
} tree_node_t;

typedef struct split_output_t
{
  int cut_feature;
  double cut_value;
} split_output_t;

// function for tree construction. xtrain and ytrain are both in cpu memory
tree_node_t *build_cart_serial(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train, int depth);

// function for making predictions on a dataset and computing accuracy. xtest
// and ytest are both in cpu memory
double eval_serial(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree);

#endif