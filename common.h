#ifndef __COMMON_H__
#define __COMMON_H__

#define DEPTH 50
#define NUM_BLOCKS 1
#define NUM_THREADS 1

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

// function for splitting. xtrain and ytrain are both in cpu memory
split_output_t split_serial(int D, int N, double *x_train, double *y_train);

// function for tree construction. xtrain and ytrain are both in cpu memory
tree_node_t* build_cart_serial(int D, int N, double *x_train, double *y_train);

// function for making predictions on a dataset and computing accuracy. xtest
// and ytest are both in cpu memory
void eval_serial(int D, int N, double *x_test, double *y_test, double *accuracy, tree_node_t *tree);

#endif