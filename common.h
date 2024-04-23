#define DEPTH 50

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

// function for splitting
split_output_t split(int d, int n, double *x_train, double *y_train);

// function for tree construction
void build_cart(int d, int n, double *x_train, double *y_train);

// function for making predictions on a dataset and computing accuracy
void eval(int d, int n, double *x_test, double *y_test, double *accuracy);
