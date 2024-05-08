#include <cuda.h>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include "common.h"
#include <cassert>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <random>

// GLOBAL VARIABLES
std::vector<double> x_train_global;
std::vector<double> y_train_global;
std::vector<double> x_test_global;
std::vector<double> y_test_global;
int train_size_global;
int test_size_global;
int D_global;

std::vector<double> predictions_global;

// HELPER FUNCTIONS

std::vector<int> idif(std::vector<double> &x, int D, int N)
{
    std::vector<int> output;

    double prevValue = x[0];
    for (int i = 1; i < N; ++i)
    {
        if (std::fabs(x[i] - prevValue) > THRESHOLD)
        {
            output.push_back(i - 1);
            prevValue = x[i];
        }
    }

    return output;
}

std::vector<int> argsort(std::vector<double> &x, int d, int D, int N)
{
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [x, D, d](int i1, int i2) -> bool
                     { return x[i1 * D + d] < x[i2 * D + d]; });
    return idx;
}

double sum(std::vector<double> &x, int start, int end_inclusive)
{
    return std::accumulate(x.begin() + start, x.begin() + end_inclusive + 1, 0.0);
}

std::vector<double> sort_by_indices_flattened(std::vector<double> &x, std::vector<int> &indices, int d, int D, int N)
{
    std::vector<double> result(N);
    int i = 0;

    for (int index : indices)
    {
        result[i] = x[index * D + d];
        i++;
    }
    return result;
}

std::vector<double> sort_by_indices(std::vector<double> &y, std::vector<int> &indices, int N)
{
    std::vector<double> result(N);

    for (int i = 0; i < N; ++i)
    {
        result[i] = y[indices[i]];
    }
    return result;
}

split_output_t split_serial(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train)
{
    double weight = 1.0 / N;
    double min_loss = std::numeric_limits<double>::infinity();
    int feature = std::numeric_limits<int>::infinity();
    double cut_value = std::numeric_limits<double>::infinity();

    // iterate through each feature
    for (int d = 0; d < D; ++d)
    {
        std::vector<int> indices = argsort(x_train, d, D, N);
        std::vector<double> x_train_sorted = sort_by_indices_flattened(x_train, indices, d, D, N);
        std::vector<double> y_train_sorted = sort_by_indices(y_train, indices, N);
        std::vector<double> y_train_sorted_squared(N);
        std::transform(y_train_sorted.begin(), y_train_sorted.end(), y_train_sorted_squared.begin(),
                       [](double x) -> double
                       { return x * x; });

        double mean_square_left = 0.0;
        double mean_left = 0.0;
        double weight_left = 0.0;
        double mean_square_right = weight * std::accumulate(y_train_sorted_squared.begin(), y_train_sorted_squared.end(), 0.0);
        double mean_right = weight * std::accumulate(y_train_sorted.begin(), y_train_sorted.end(), 0.0);
        double weight_right = 1.0;

        // Only consider splits with at least one value on each side
        std::vector<int> idif_idx = idif(x_train_sorted, D, N);

        int pi = 0;
        for (int i : idif_idx)
        {
            double delta_mean_squared = weight * sum(y_train_sorted_squared, pi, i);
            double delta_mean = weight * sum(y_train_sorted, pi, i);

            mean_square_left += delta_mean_squared;
            mean_left += delta_mean;

            weight_left += weight * (i + 1 - pi);
            assert((i + 1 - pi) != 0);

            mean_square_right -= delta_mean_squared;
            mean_right -= delta_mean;
            weight_right -= weight * (i + 1 - pi);

            double left_loss = mean_square_left - (mean_left * mean_left) / weight_left;
            double right_loss = mean_square_right - (mean_right * mean_right) / weight_right;
            double loss = left_loss + right_loss;

            if (loss < min_loss)
            {
                min_loss = loss;
                feature = d;
                cut_value = (x_train_sorted[i] + x_train_sorted[i + 1]) / 2;
            }

            pi = i + 1;
        }
    }

    assert(feature != std::numeric_limits<int>::infinity());

    split_output_t output;
    output.cut_feature = feature;
    output.cut_value = cut_value;
    output.loss = min_loss;
    return output;
}

/** Checks that all elements in a vector are equal to a value within some error */
bool elements_equal(std::vector<double> &values, int size, double value, double epsilon)
{
    for (int i = 0; i < size; ++i)
    {
        if (std::fabs(values[i] - value) > epsilon)
        {
            return false;
        }
    }
    return true;
}

/** Checks that all rows are equal within some error*/
bool rows_equal(std::vector<double> &x, int D, int N, double epsilon)
{
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            if (std::fabs(x[i * D + j] - x[(i + 1) * D + j]) > epsilon)
            {
                return false;
            }
        }
    }
    return true;
}

tree_node_t *build_cart_helper(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train, int depth)
{
    double mean = 0.0;
    for (int i = 0; i < N; ++i)
    {
        mean += y_train[i];
    }

    mean /= N;

    // if no more branching can be done, return a leaf node
    if (depth == 0 || elements_equal(y_train, N, y_train[0], THRESHOLD) || rows_equal(x_train, D, N, THRESHOLD))
    {
        tree_node_t *leaf = (tree_node_t *)malloc(sizeof(tree_node_t));
        leaf->left = NULL;
        leaf->right = NULL;
        leaf->parent = NULL;
        leaf->prediction = mean;
        leaf->cut_feature = -1;
        leaf->cut_value = NAN;
        return leaf;
    }
    else
    {
        split_output_t split = split_serial(D, N, x_train, y_train);

        std::vector<double> left_x_train, right_x_train;
        std::vector<double> left_y_train, right_y_train;

        for (int i = 0; i < N; ++i)
        {
            double x = x_train[i * D + split.cut_feature];
            double y = y_train[i];

            if (x <= split.cut_value)
            {
                left_x_train.insert(left_x_train.end(), x_train.begin() + i * D, x_train.begin() + i * D + D);
                left_y_train.push_back(y);
            }
            else
            {
                right_x_train.insert(right_x_train.end(), x_train.begin() + i * D, x_train.begin() + i * D + D);
                right_y_train.push_back(y);
            }
        }

        // recursively build left and right subtrees
        tree_node_t *left = build_cart_helper(D, left_y_train.size(), left_x_train, left_y_train, depth - 1);
        tree_node_t *right = build_cart_helper(D, right_y_train.size(), right_x_train, right_y_train, depth - 1);

        tree_node_t *node = (tree_node_t *)malloc(sizeof(tree_node_t));
        node->cut_feature = split.cut_feature;
        node->cut_value = split.cut_value;
        node->left = left;
        node->right = right;
        node->prediction = mean;
        left->parent = node;
        right->parent = node;
        return node;
    }
}

// IMPLEMENTATION

void init(int D, int train_size, int test_size, std::vector<double> &x_train, std::vector<double> &y_train, std::vector<double> &x_test, std::vector<double> &y_test)
{
    D_global = D;
    train_size_global = train_size;
    test_size_global = test_size;
    x_train_global = x_train;
    y_train_global = y_train;
    x_test_global = x_test;
    y_test_global = y_test;
    predictions_global = std::vector<double>(test_size_global);
}

tree_node_t *build_cart(int depth)
{
    return build_cart_helper(D_global, train_size_global, x_train_global, y_train_global, depth);
}

tree_node_t *build_cart_iterative(int depth)
{
    std::vector<tree_node_t *> tree;
    using XYpair = std::pair<std::vector<double>, std::vector<double>>;

    // Pair of (x training data, y training data)
    std::vector<XYpair> split_data_curr(1);
    std::vector<XYpair> split_data_temp;
    split_data_curr[0] = XYpair(x_train_global, y_train_global);

    int current_layer_idx = 0;
    for (int i = 0; i <= depth; ++i)
    {
        int curr_level_size = pow(2, i);
        std::vector<bool> threadFinished(curr_level_size, true);
        std::vector<tree_node_t *> curr_level(curr_level_size);

        split_data_temp.resize(pow(2, i + 1));

        for (int j = 0; j < curr_level_size; ++j)
        {
            int current_idx = current_layer_idx + j;
            std::vector<double> x_train_curr;
            std::vector<double> y_train_curr;

            x_train_curr = split_data_curr[j].first;
            y_train_curr = split_data_curr[j].second;

            int N = y_train_curr.size();

            double weight = 1.0 / N;
            double mean = 0.0;
            for (int k = 0; k < N; ++k)
            {
                mean += y_train_curr[k];
            }

            mean /= N;

            int parent_idx = (current_idx - 1) / 2;
            bool isLeftChild = (current_idx - 1) % 2 == 0;

            // Case where there should be no node at this index
            // If the parent node is a leaf or the parent node is also null, we push back null
            if (current_idx != 0 && (tree[parent_idx] == NULL || tree[parent_idx]->cut_feature == -1))
            {
                curr_level[j] = NULL;
                continue;
            }
            // Else if the current node's parent is not a leaf but the stopping criteria is reached, push back leaf
            else if (i == depth || elements_equal(y_train_curr, N, y_train_curr[0], THRESHOLD) || rows_equal(x_train_curr, D_global, N, THRESHOLD))
            {
                tree_node_t *leaf = (tree_node_t *)malloc(sizeof(tree_node_t));
                leaf->left = NULL;
                leaf->right = NULL;
                leaf->parent = NULL;
                leaf->prediction = mean;
                leaf->cut_feature = -1;
                leaf->cut_value = NAN;
                curr_level[j] = leaf;

                if (tree[parent_idx]->cut_feature != -1)
                {
                    if (isLeftChild)
                    {
                        tree[parent_idx]->left = leaf;
                    }
                    else
                    {
                        tree[parent_idx]->right = leaf;
                    }
                }

                continue;
            }

            threadFinished[j] = false;

            split_output_t split = split_serial(D_global, N, x_train_curr, y_train_curr);

            std::vector<double> left_x_train, right_x_train;
            std::vector<double> left_y_train, right_y_train;

            for (int i = 0; i < N; ++i)
            {
                double x = x_train_curr[i * D_global + split.cut_feature];
                double y = y_train_curr[i];

                if (x <= split.cut_value)
                {
                    left_x_train.insert(left_x_train.end(), x_train_curr.begin() + i * D_global, x_train_curr.begin() + i * D_global + D_global);
                    left_y_train.push_back(y);
                }
                else
                {
                    right_x_train.insert(right_x_train.end(), x_train_curr.begin() + i * D_global, x_train_curr.begin() + i * D_global + D_global);
                    right_y_train.push_back(y);
                }
            }

            // Insert data for the next level
            split_data_temp[j * 2] = XYpair(left_x_train, left_y_train);
            split_data_temp[j * 2 + 1] = XYpair(right_x_train, right_y_train);

            tree_node_t *node = (tree_node_t *)malloc(sizeof(tree_node_t));
            node->cut_feature = split.cut_feature;
            node->cut_value = split.cut_value;
            node->left = NULL;
            node->right = NULL;
            node->prediction = mean;

            if (current_idx != 0)
            {
                if (isLeftChild)
                {
                    tree[parent_idx]->left = node;
                }
                else
                {
                    tree[parent_idx]->right = node;
                }
                node->parent = tree[parent_idx];
            }
            curr_level[j] = node;
        }

        // update the starting index of the next layer
        current_layer_idx += curr_level_size;

        tree.insert(tree.end(), curr_level.begin(), curr_level.end());

        // Check if the entire layer is NULL or Leaves
        bool allTrue = std::all_of(threadFinished.begin(), threadFinished.end(), [](bool element)
                                   { return element; });
        if (allTrue)
        {
            break;
        }

        // populate the split data for the next layer
        split_data_curr = split_data_temp;
        split_data_temp.clear();
    }
    return tree[0];
}

/** Recursive helper for evaluating an input data point using the root */
double eval_helper(tree_node_t *tree, std::vector<double> &data)
{
    if (tree->left == NULL && tree->right == NULL)
    {
        return tree->prediction;
    }

    int feature = tree->cut_feature;
    double cut_value = tree->cut_value;

    if (data[feature] <= cut_value)
    {
        return eval_helper(tree->left, data);
    }
    else
    {
        return eval_helper(tree->right, data);
    }
}

void compute_predictions(tree_node_t *tree)
{
    // compute predictions
    for (int i = 0; i < test_size_global; ++i)
    {
        std::vector<double> data = std::vector<double>(x_test_global.begin() + i * D_global, x_test_global.begin() + i * D_global + D_global);
        double prediction = eval_helper(tree, data);
        predictions_global[i] = prediction;
    }
}

double eval_mse(tree_node_t *tree)
{
    compute_predictions(tree);
    // compute MSE
    double accumulator = 0;
    for (int i = 0; i < test_size_global; ++i)
    {
        accumulator += pow(predictions_global[i] - y_test_global[i], 2);
    }

    return accumulator / test_size_global;
}

double eval_classification(tree_node_t *tree)
{
    compute_predictions(tree);

    // compute classification error
    double accumulator = 0;
    for (int i = 0; i < test_size_global; ++i)
    {
        if (predictions_global[i] != y_test_global[i])
        {
            accumulator += 1;
        }
    }

    return accumulator / test_size_global;
}

forest_t *build_forest(int depth, int num_trees)
{
    int subsample_size = ceil(SUBSAMPLE_RATE * train_size_global);
    forest_t *trees = new std::vector<tree_node_t *>(num_trees);
    std::vector<double> x_train_rand(subsample_size * D_global);
    std::vector<double> y_train_rand(subsample_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, train_size_global - 1);

    int random_idx;
    tree_node_t *curr_tree;
    for (int i = 0; i < num_trees; i++)
    {
        for (int j = 0; j < subsample_size; j++)
        {
            random_idx = dist(gen);
            for (int k = 0; k < D_global; k++)
            {
                x_train_rand[j * D_global + k] = x_train_global[random_idx * D_global + k];
            }
            y_train_rand[j] = y_train_global[random_idx];
        }
        curr_tree = build_cart_helper(D_global, subsample_size, x_train_rand, y_train_rand, depth);
        (*trees)[i] = curr_tree;
    }

    return trees;
}

double eval_forest_mse(forest_t *forest)
{
    double weight = 1.0 / (*forest).size();

    for (int i = 0; i < (*forest).size(); i++)
    {
        for (int j = 0; j < test_size_global; j++)
        {
            std::vector<double> data = std::vector<double>(x_test_global.begin() + j * D_global, x_test_global.begin() + j * D_global + D_global);
            double prediction = eval_helper((*forest)[i], data);
            predictions_global[j] += weight * prediction;
        }
    }

    // compute MSE
    double accumulator = 0;
    for (int i = 0; i < test_size_global; ++i)
    {
        accumulator += pow(predictions_global[i] - y_test_global[i], 2);
    }

    return accumulator / test_size_global;
}