#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/adjacent_difference.h>
#include <thrust/count.h>
#include <thrust/tuple.h>
#include <vector>
#include "common.h"
#include "util.h"
#include <utility>

__device__ int round_down(int x, int D)
{
    return x - (x % D);
}

thrust::device_vector<int> idif_gpu(thrust::device_vector<double> &x, int N)
{
    // Find differences between adjacent values and store in diffs vector
    thrust::device_vector<double> adj_diffs(N);
    thrust::adjacent_difference(x.begin(), x.end(), adj_diffs.begin());

    // Make a zipped iterator that iterates diffs and idx
    thrust::device_vector<int> idx(N);
    thrust::sequence(idx.begin(), idx.end());
    thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator,
                                       thrust::device_vector<double>::iterator>>
        zipped_iterator = thrust::make_zip_iterator(thrust::make_tuple(idx.begin(), adj_diffs.begin()));

    // Count_if the number of elements that are different w.r.t. threshold
    int size = thrust::count_if(zipped_iterator, zipped_iterator + idx.size(), [] __device__(thrust::tuple<int, double> pair)
                                { return thrust::get<1>(pair) > THRESHOLD && thrust::get<0>(pair) != 0; });
    thrust::device_vector<thrust::tuple<int, double>> idif_tuples(size);

    // Use copy_if to store indices
    thrust::copy_if(
        zipped_iterator, zipped_iterator + idx.size(), idif_tuples.begin(), [] __device__(thrust::tuple<int, double> pair)
        { return thrust::get<1>(pair) > THRESHOLD && thrust::get<0>(pair) != 0; });

    // Use transform to subtract 1 from indices
    thrust::device_vector<int> idif_idx(size);
    thrust::transform(idif_tuples.begin(), idif_tuples.end(), idif_idx.begin(), [] __device__(thrust::tuple<int, double> pair)
                      { return thrust::get<0>(pair) - 1; });

    return idif_idx;
}

thrust::device_vector<int> argsort_exhaustive_gpu(thrust::device_vector<double> &x, int d, int D, int N)
{
    thrust::device_vector<int> indices(D * N);
    thrust::sequence(indices.begin(), indices.end());

    double *x_ptr = thrust::raw_pointer_cast(x.data());

    thrust::stable_sort(indices.begin(), indices.end(), [x_ptr, D, d] __device__(int left_idx, int right_idx)
                        { return x_ptr[(left_idx - (left_idx % D)) + d] < x_ptr[right_idx - (right_idx % D) + d]; });
    return indices;
}

thrust::device_vector<int> argsort_gpu(thrust::device_vector<double> &x, int d, int D, int N)
{
    thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end());

    double *x_ptr = thrust::raw_pointer_cast(x.data());

    thrust::stable_sort(indices.begin(), indices.end(), [x_ptr, D, d] __device__(int left_idx, int right_idx)
                        { return x_ptr[left_idx * D + d] < x_ptr[right_idx * D + d]; });
    return indices;
}

split_output_t split_gpu(int D, int N, thrust::device_vector<double> &x_train, thrust::device_vector<double> &y_train)
{
    double weight = 1.0 / N;

    thrust::device_vector<split_output_t> split_results((N - 1) * D);

    split_output_t init_output;
    init_output.cut_feature = std::numeric_limits<int>::infinity();
    init_output.cut_value = std::numeric_limits<double>::infinity();
    init_output.loss = std::numeric_limits<double>::infinity();
    thrust::fill(split_results.begin(), split_results.end(), init_output);

    // iterate through each feature
    for (int d = 0; d < D; ++d)
    {
        thrust::device_vector<int> indices_y = argsort_gpu(x_train, d, D, N);

        thrust::device_vector<int> indices_x(N);
        thrust::transform(indices_y.begin(), indices_y.end(), indices_x.begin(),
                          [D, d] __device__(int y_idx) -> int
                          { return y_idx * D + d; });

        thrust::device_vector<double> x_train_sorted(N);
        thrust::device_vector<double> y_train_sorted(N);

        thrust::gather(thrust::device, indices_x.begin(), indices_x.end(), x_train.begin(), x_train_sorted.begin());
        thrust::gather(thrust::device, indices_y.begin(), indices_y.end(), y_train.begin(), y_train_sorted.begin());

        thrust::device_vector<double> y_train_sorted_squared(N);
        thrust::transform(y_train_sorted.begin(), y_train_sorted.end(), y_train_sorted_squared.begin(),
                          [] __device__(double x) -> double
                          { return x * x; });

        double mean_square_right = weight * thrust::reduce(y_train_sorted_squared.begin(), y_train_sorted_squared.end(), 0.0);
        double mean_right = weight * thrust::reduce(y_train_sorted.begin(), y_train_sorted.end(), 0.0);

        thrust::device_vector<double> y_prefix_sum(N);
        thrust::device_vector<double> y_squared_prefix_sum(N);

        thrust::inclusive_scan(thrust::device, y_train_sorted.begin(), y_train_sorted.end(), y_prefix_sum.begin());
        thrust::inclusive_scan(thrust::device, y_train_sorted_squared.begin(), y_train_sorted_squared.end(), y_squared_prefix_sum.begin());

        thrust::device_vector<int> idif_indices = idif_gpu(x_train_sorted, N);

        split_output_t *split_results_ptr = thrust::raw_pointer_cast(split_results.data());
        double *x_train_sorted_ptr = thrust::raw_pointer_cast(x_train_sorted.data());

        double *y_prefix_sum_ptr = thrust::raw_pointer_cast(y_prefix_sum.data());
        double *y_squared_prefix_sum_ptr = thrust::raw_pointer_cast(y_squared_prefix_sum.data());

        // todo go through idif indices
        thrust::for_each(idif_indices.begin(), idif_indices.end(),
                         [N, D, d, weight, mean_square_right, mean_right, split_results_ptr, y_prefix_sum_ptr, y_squared_prefix_sum_ptr, x_train_sorted_ptr] __device__(int index)
                         {
                             double mean_square_left = weight * y_squared_prefix_sum_ptr[index];
                             double mean_left = weight * y_prefix_sum_ptr[index];
                             double local_mean_right = mean_right - mean_left;
                             double local_mean_sq_right = mean_square_right - mean_square_left;
                             double weight_left = (index + 1) * weight;
                             double weight_right = (N - index - 1) * weight;
                             double left_loss = mean_square_left - (mean_left * mean_left) / weight_left;
                             double right_loss = local_mean_sq_right - (local_mean_right * local_mean_right) / weight_right;
                             split_output_t split_output;
                             split_output.cut_feature = d;
                             split_output.cut_value = (x_train_sorted_ptr[index] + x_train_sorted_ptr[index + 1]) / 2;
                             split_output.loss = left_loss + right_loss;
                             split_results_ptr[(N - 1) * d + index] = split_output;
                         });

        cudaDeviceSynchronize();
    }

    split_output_t output = thrust::reduce(thrust::device, split_results.begin(), split_results.end(), init_output,
                                           [] __device__ __host__(split_output_t left, split_output_t right)
                                           {
                                               if (left.loss < right.loss)
                                               {
                                                   return left;
                                               }
                                               else
                                               {
                                                   return right;
                                               }
                                           });

    // std::cout << "split feature: " << output.cut_feature << " split value: " << output.cut_value << " loss: " << output.loss << std::endl;

    return output;
}

/** Checks that all elements in a vector are equal to a value within some error */
bool elements_equal_gpu(std::vector<double> &values, int size, double value, double epsilon)
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
bool rows_equal_gpu(std::vector<double> &x, int D, int N, double epsilon)
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

tree_node_t *build_cart(int D, int N_initial, std::vector<double> &x_train, std::vector<double> &y_train, int depth)
{
    std::vector<tree_node_t *> tree;
    using XYpair = std::pair<std::vector<double>, std::vector<double>>;

    // Pair of (x training data, y training data)
    std::vector<XYpair> split_data_curr(1);
    std::vector<XYpair> split_data_temp;
    split_data_curr[0] = XYpair(x_train, y_train);

    int current_idx = 0;
    for (int i = 0; i <= depth; ++i)
    {
        bool allNull = true;
        split_data_temp.resize(pow(2, i + 1));
        for (int j = 0; j < pow(2, i); ++j)
        {
            std::vector<double> x_train_curr;
            std::vector<double> y_train_curr;
            // First iteration contains entirety of training data

            x_train_curr = split_data_curr[j].first;
            y_train_curr = split_data_curr[j].second;
            int N = y_train_curr.size();

            // std::cout << "N " << N << " x_train size: " << x_train_curr.size() << std::endl;
            double weight = 1.0 / N;
            double mean = 0.0;
            for (int i = 0; i < N; ++i)
            {
                mean += y_train_curr[i];
            }

            mean /= N;

            int parent_idx = (current_idx - 1) / 2;
            bool isLeftChild = (current_idx - 1) % 2 == 0;

            // Case where there should be no node at this index
            // If the parent node is a leaf or the parent node is also null, we push back null
            if (current_idx != 0 && (tree[parent_idx] == NULL || tree[parent_idx]->cut_feature == -1))
            {
                tree.push_back(NULL);
                current_idx += 1;
                continue;
            }
            // Else if the current node's parent is not a leaf but the stopping criteria is reached, push back leaf
            else if (i == depth || elements_equal_gpu(y_train_curr, N, y_train_curr[0], THRESHOLD) || rows_equal_gpu(x_train_curr, D, N, THRESHOLD))
            {
                tree_node_t *leaf = (tree_node_t *)malloc(sizeof(tree_node_t));
                leaf->left = NULL;
                leaf->right = NULL;
                leaf->parent = NULL;
                leaf->prediction = mean;
                leaf->cut_feature = -1;
                leaf->cut_value = NAN;
                tree.push_back(leaf);

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

                current_idx += 1;
                continue;
            }

            allNull = false;

            thrust::device_vector<double> d_x_train(x_train_curr.begin(), x_train_curr.end());
            thrust::device_vector<double> d_y_train(y_train_curr.begin(), y_train_curr.end());

            // std::cout << "d_x_train size: " << d_x_train.size() << "d_y_train size: " << d_y_train.size() << std::endl;

            split_output_t split = split_gpu(D, N, d_x_train, d_y_train);
            // std::cout << "split feature: " << split.cut_feature << "split value: " << split.cut_value << "loss: " << split.loss << std::endl;

            std::vector<double> left_x_train, right_x_train;
            std::vector<double> left_y_train, right_y_train;

            for (int i = 0; i < N; ++i)
            {
                double x = x_train_curr[i * D + split.cut_feature];
                double y = y_train_curr[i];

                if (x <= split.cut_value)
                {
                    left_x_train.insert(left_x_train.end(), x_train_curr.begin() + i * D, x_train_curr.begin() + i * D + D);
                    left_y_train.push_back(y);
                }
                else
                {
                    right_x_train.insert(right_x_train.end(), x_train_curr.begin() + i * D, x_train_curr.begin() + i * D + D);
                    right_y_train.push_back(y);
                }
            }
            // std::cout << "Created training splits" << std::endl;

            // Insert data for the next level
            split_data_temp[j * 2] = XYpair(left_x_train, left_y_train);
            split_data_temp[j * 2 + 1] = XYpair(right_x_train, right_y_train);

            // std::cout << "Insert training data into split data temp" << std::endl;

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
            tree.push_back(node);

            current_idx += 1;
        }

        // Check if the entire layer is NULL or Leaves
        if (allNull)
        {
            break;
        }

        split_data_curr = split_data_temp;
        split_data_temp.clear();
    }
    return tree[0];
}

/** Recursive helper for evaluating an input data point using a tree */
double eval_helper_gpu(tree_node_t *tree, std::vector<double> &data)
{
    if (tree->left == NULL && tree->right == NULL)
    {
        return tree->prediction;
    }

    int feature = tree->cut_feature;
    double cut_value = tree->cut_value;

    if (data[feature] <= cut_value)
    {
        return eval_helper_gpu(tree->left, data);
    }
    else
    {
        return eval_helper_gpu(tree->right, data);
    }
}

double eval_mse(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree)
{
    // compute predictions
    std::vector<double> predictions(N);
    for (int i = 0; i < N; ++i)
    {
        std::vector<double> data = std::vector<double>(x_test.begin() + i * D, x_test.begin() + i * D + D);
        double prediction = eval_helper_gpu(tree, data);
        predictions[i] = prediction;
    }

    if (WRITE_TO_CSV)
    {
        write_data_to_csv("../datasets/pred.csv", predictions, N, 1);
    }

    // compute MSE
    double accumulator = 0;
    for (int i = 0; i < N; ++i)
    {
        accumulator += pow(predictions[i] - y_test[i], 2);
    }

    return accumulator / N;
}

double eval_classification(int D, int N, std::vector<double> &x_test, std::vector<double> &y_test, tree_node_t *tree)
{
    // compute predictions
    std::vector<double> predictions(N);
    for (int i = 0; i < N; ++i)
    {
        std::vector<double> data = std::vector<double>(x_test.begin() + i * D, x_test.begin() + i * D + D);
        double prediction = eval_helper_gpu(tree, data);
        predictions[i] = prediction;
    }

    if (WRITE_TO_CSV)
    {
        write_data_to_csv("../datasets/pred.csv", predictions, N, 1);
    }

    // compute MSE
    double accumulator = 0;
    for (int i = 0; i < N; ++i)
    {
        accumulator += pow(predictions[i] - y_test[i], 2);
    }

    return accumulator / N;
}