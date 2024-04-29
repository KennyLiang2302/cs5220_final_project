#include <cuda.h>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include "common.h"
#include <cassert>
#include <iostream>

std::vector<int> argsort(std::vector<double> &x, int d, int D, int N)
{
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [x, D, d](int i1, int i2) -> bool
                     { return x[i1 * D + d] < x[i2 * D + d]; });
    return idx;
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
        // TODO: We should only consider splits between two different numbers
        for (int i = 0; i < N - 1; ++i)
        {
            double delta_mean_squared = weight * y_train_sorted_squared[i];
            double delta_mean = weight * y_train_sorted[i];

            mean_square_left += delta_mean_squared;
            mean_left += delta_mean;
            weight_left += weight;

            mean_square_right -= delta_mean_squared;
            mean_right -= delta_mean;
            weight_right -= weight;

            double left_loss = mean_square_left - pow(mean_left, 2) / weight_left;
            double right_loss = mean_square_right - pow(mean_right, 2) / weight_right;
            double loss = left_loss + right_loss;

            if (loss < min_loss)
            {
                min_loss = loss;
                feature = d;
                cut_value = (x_train_sorted[i] + x_train_sorted[i + 1]) / 2;
            }
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

tree_node_t *build_cart(int D, int N, std::vector<double> &x_train, std::vector<double> &y_train, int depth)
{
    double weight = 1.0 / N;
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
        if (right_y_train.size() == 0 || left_y_train.size() == 0)
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

        // recursively build left and right subtrees
        tree_node_t *left = build_cart(D, left_y_train.size(), left_x_train, left_y_train, depth - 1);
        tree_node_t *right = build_cart(D, right_y_train.size(), right_x_train, right_y_train, depth - 1);

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

/** Recursive helper for evaluating an input data point using a tree */
double eval_helper(tree_node_t *tree, std::vector<double> &data)
{
    if (tree->left == NULL && tree->right == NULL)
    {
        return tree->prediction;
    }

    int feature = tree->cut_feature;
    int cut_value = tree->cut_value;

    if (data[feature] <= cut_value)
    {
        return eval_helper(tree->left, data);
    }
    else
    {
        return eval_helper(tree->right, data);
    }
}

double eval(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree)
{
    // compute predictions
    std::vector<double> predictions(N);
    for (int i = 0; i < N; ++i)
    {
        std::vector<double> data = std::vector<double>(x_test.begin() + i * D, x_test.begin() + i * D + D);
        double prediction = eval_helper(tree, data);
        predictions[i] = prediction;
    }

    // compute MSE
    double accumulator = 0;
    for (int i = 0; i < N; ++i)
    {
        accumulator += pow(predictions[i] - y_test[i], 2);
    }

    return accumulator / N;
}
