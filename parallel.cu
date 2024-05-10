#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/adjacent_difference.h>
#include <thrust/count.h>
#include <thrust/tuple.h>
#include <thrust/logical.h>
#include <vector>
#include "common.h"
#include <random>
#include <thrust/random.h>
#include <omp.h>
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

// GLOBAL VARIABLES
thrust::device_vector<double> *x_train_global;
thrust::device_vector<double> *y_train_global;
std::vector<double> x_test_global;
std::vector<double> y_test_global;
int train_size_global;
int test_size_global;
int D_global;

std::vector<double> predictions_global;

// HELPER FUNCTIONS

__device__ int round_down(int x, int D)
{
    return x - (x % D);
}

// TODO, can try to optimize this function.
template <typename DerivedPolicy>
thrust::device_vector<int> idif_gpu(const thrust::detail::execution_policy_base<DerivedPolicy> &exec_policy, thrust::device_vector<double> &x, int N)
{
    // Find differences between adjacent values and store in diffs vector
    thrust::device_vector<double> adj_diffs(N);
    thrust::adjacent_difference(exec_policy, x.begin(), x.end(), adj_diffs.begin());

    // Make a zipped iterator that iterates diffs and idx
    thrust::device_vector<int> idx(N);
    thrust::sequence(exec_policy, idx.begin(), idx.end());
    thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator,
                                       thrust::device_vector<double>::iterator>>
        zipped_iterator = thrust::make_zip_iterator(thrust::make_tuple(idx.begin(), adj_diffs.begin()));

    // Count_if the number of elements that are different w.r.t. threshold
    int size = thrust::count_if(exec_policy, zipped_iterator, zipped_iterator + idx.size(), [] __device__(thrust::tuple<int, double> pair)
                                { return thrust::get<1>(pair) > THRESHOLD && thrust::get<0>(pair) != 0; });
    thrust::device_vector<thrust::tuple<int, double>> idif_tuples(size);

    // Use copy_if to store indices
    thrust::copy_if(exec_policy,
                    zipped_iterator, zipped_iterator + idx.size(), idif_tuples.begin(), [] __device__(thrust::tuple<int, double> pair)
                    { return thrust::get<1>(pair) > THRESHOLD && thrust::get<0>(pair) != 0; });

    // Use transform to subtract 1 from indices
    thrust::device_vector<int> idif_idx(size);
    thrust::transform(exec_policy, idif_tuples.begin(), idif_tuples.end(), idif_idx.begin(), [] __device__(thrust::tuple<int, double> pair)
                      { return thrust::get<0>(pair) - 1; });

    return idif_idx;
}

template <typename DerivedPolicy>
thrust::device_vector<int> argsort_gpu(const thrust::detail::execution_policy_base<DerivedPolicy> &exec_policy, thrust::device_vector<double> &x, int d, int D, int N)
{
    thrust::device_vector<int> indices(N);
    thrust::sequence(exec_policy, indices.begin(), indices.end());

    double *x_ptr = thrust::raw_pointer_cast(x.data());

    thrust::stable_sort(exec_policy, indices.begin(), indices.end(), [x_ptr, D, d] __device__(int left_idx, int right_idx)
                        { return x_ptr[left_idx * D + d] < x_ptr[right_idx * D + d]; });
    return indices;
}

template <typename DerivedPolicy>
split_output_t split_gpu(const thrust::detail::execution_policy_base<DerivedPolicy> &exec_policy, int D, int N, thrust::device_vector<double> &x_train, thrust::device_vector<double> &y_train)
{
    double weight = 1.0 / N;

    thrust::device_vector<split_output_t> split_results((N - 1) * D);

    split_output_t init_output;
    init_output.cut_feature = std::numeric_limits<int>::infinity();
    init_output.cut_value = std::numeric_limits<double>::infinity();
    init_output.loss = std::numeric_limits<double>::infinity();
    thrust::fill(exec_policy, split_results.begin(), split_results.end(), init_output);

    // iterate through each feature
    for (int d = 0; d < D; ++d)
    {
        thrust::device_vector<int> indices_y = argsort_gpu(exec_policy, x_train, d, D, N);

        thrust::device_vector<int> indices_x(N);
        thrust::transform(exec_policy, indices_y.begin(), indices_y.end(), indices_x.begin(),
                          [D, d] __device__(int y_idx) -> int
                          { return y_idx * D + d; });

        thrust::device_vector<double> x_train_sorted(N);
        thrust::device_vector<double> y_train_sorted(N);

        thrust::gather(exec_policy, indices_x.begin(), indices_x.end(), x_train.begin(), x_train_sorted.begin());
        thrust::gather(exec_policy, indices_y.begin(), indices_y.end(), y_train.begin(), y_train_sorted.begin());

        thrust::device_vector<double> y_train_sorted_squared(N);
        thrust::transform(exec_policy, y_train_sorted.begin(), y_train_sorted.end(), y_train_sorted_squared.begin(),
                          [] __device__(double x) -> double
                          { return x * x; });

        double mean_square_right = weight * thrust::reduce(exec_policy, y_train_sorted_squared.begin(), y_train_sorted_squared.end(), 0.0);
        double mean_right = weight * thrust::reduce(exec_policy, y_train_sorted.begin(), y_train_sorted.end(), 0.0);

        thrust::device_vector<double> y_prefix_sum(N);
        thrust::device_vector<double> y_squared_prefix_sum(N);

        thrust::inclusive_scan(exec_policy, y_train_sorted.begin(), y_train_sorted.end(), y_prefix_sum.begin());
        thrust::inclusive_scan(exec_policy, y_train_sorted_squared.begin(), y_train_sorted_squared.end(), y_squared_prefix_sum.begin());

        thrust::device_vector<int> idif_indices = idif_gpu(exec_policy, x_train_sorted, N);

        split_output_t *split_results_ptr = thrust::raw_pointer_cast(split_results.data());
        double *x_train_sorted_ptr = thrust::raw_pointer_cast(x_train_sorted.data());

        double *y_prefix_sum_ptr = thrust::raw_pointer_cast(y_prefix_sum.data());
        double *y_squared_prefix_sum_ptr = thrust::raw_pointer_cast(y_squared_prefix_sum.data());

        // todo go through idif indices
        thrust::for_each(exec_policy, idif_indices.begin(), idif_indices.end(),
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

    split_output_t output = thrust::reduce(exec_policy, split_results.begin(), split_results.end(), init_output,
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

    return output;
}

/** Checks that all elements in a vector are equal to a value within some error
 */
template <typename DerivedPolicy>
bool elements_equal_gpu(const thrust::detail::execution_policy_base<DerivedPolicy> &exec_policy, thrust::device_vector<double> &values, int size,
                        double epsilon)
{
    thrust::device_ptr<double> values_ptr = values.data();
    return thrust::count_if(exec_policy, values.begin(), values.end(),
                            [=] __device__(double x)
                            {
                                return fabs(x - values_ptr[0]) > epsilon;
                            }) == 0;
}

/** Checks that all rows are equal within some error*/
template <typename DerivedPolicy>
bool rows_equal_gpu(const thrust::detail::execution_policy_base<DerivedPolicy> &exec_policy, thrust::device_vector<double> &d_x, int D, int N, double epsilon)
{
    // TODO, same as above
    thrust::device_ptr<double> d_x_ptr = d_x.data();

    return thrust::count_if(exec_policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N - 1),
                            [=] __device__(int i)
                            {
                                for (int j = 0; j < D; ++j)
                                {
                                    if (fabs(d_x_ptr[i * D + j] - d_x_ptr[(i + 1) * D + j]) > epsilon)
                                    {
                                        return true;
                                    }
                                }
                                return false;
                            }) == 0;
}

tree_node_t *build_cart_helper(int D, int N, thrust::device_vector<double> &d_x_train, thrust::device_vector<double> &d_y_train, int depth)
{
    double weight = 1.0 / N;
    double mean = thrust::reduce(thrust::device, d_y_train.begin(), d_y_train.end(), 0.0) / N;

    // if no more branching can be done, return a leaf node
    if (depth == 0 || elements_equal_gpu(thrust::device, d_y_train, N, THRESHOLD) || rows_equal_gpu(thrust::device, d_x_train, D, N, THRESHOLD))
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
        thrust::device_ptr<double> d_x_train_ptr = d_x_train.data();

        split_output_t split = split_gpu(thrust::device, D, N, d_x_train, d_y_train);

        int size_left = thrust::count_if(thrust::device,
                                         thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                                         [d_x_train_ptr, split, D] __device__(int i)
                                         {
                                             double x = d_x_train_ptr[i * D + split.cut_feature];
                                             return x <= split.cut_value;
                                         });
        int size_right = N - size_left;

        thrust::device_vector<int> left_indices(size_left);
        thrust::device_vector<int> right_indices(size_right);
        thrust::device_vector<int> left_x_indices(size_left * D);
        thrust::device_vector<int> right_x_indices(size_right * D);

        thrust::stable_partition_copy(thrust::device,
                                      thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                                      left_indices.begin(), right_indices.begin(),
                                      [=] __device__(int i)
                                      {
                                          double x = d_x_train_ptr[i * D + split.cut_feature];
                                          return x <= split.cut_value;
                                      });

        thrust::stable_partition_copy(thrust::device,
                                      thrust::make_counting_iterator(0), thrust::make_counting_iterator(N * D),
                                      left_x_indices.begin(), right_x_indices.begin(),
                                      [=] __device__(int i)
                                      {
                                          double x = d_x_train_ptr[(i / D) * D + split.cut_feature];
                                          return x <= split.cut_value;
                                      });

        thrust::device_vector<double> left_x_train(left_indices.size() * D);
        thrust::device_vector<double> right_x_train(right_indices.size() * D);
        thrust::device_vector<double> left_y_train(left_indices.size());
        thrust::device_vector<double> right_y_train(right_indices.size());

        thrust::gather(thrust::device, left_indices.begin(), left_indices.end(), d_y_train.begin(), left_y_train.begin());
        thrust::gather(thrust::device, right_indices.begin(), right_indices.end(), d_y_train.begin(), right_y_train.begin());
        thrust::gather(thrust::device, left_x_indices.begin(), left_x_indices.end(), d_x_train.begin(), left_x_train.begin());
        thrust::gather(thrust::device, right_x_indices.begin(), right_x_indices.end(), d_x_train.begin(), right_x_train.begin());

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
    x_train_global = new thrust::device_vector<double>(x_train.begin(), x_train.end());
    y_train_global = new thrust::device_vector<double>(y_train.begin(), y_train.end());
    x_test_global = x_test;
    y_test_global = y_test;
    predictions_global = std::vector<double>(test_size_global);
}

tree_node_t *build_cart(int depth)
{
    return build_cart_helper(D_global, train_size_global, *x_train_global, *y_train_global, depth);
}

tree_node_t *build_cart_iterative(int depth)
{
    std::vector<tree_node_t *> tree;
    using XYpair = std::pair<thrust::device_vector<double>, thrust::device_vector<double>>;

    // Pair of (x training data, y training data)
    std::vector<XYpair> split_data_curr(1);
    std::vector<XYpair> split_data_temp;
    split_data_curr[0] = XYpair(*x_train_global, *y_train_global);

    omp_set_num_threads(NUM_THREADS);

    // Array of CUDA Streams
    cudaStream_t streams[NUM_STREAMS];

    for (int stream_idx = 0; stream_idx < NUM_STREAMS; ++stream_idx)
    {
        cudaStreamCreate(&streams[stream_idx]);
    }

    int current_layer_idx = 0;
    for (int i = 0; i <= depth; ++i)
    {
        int curr_level_size = pow(2, i);
        std::vector<bool> threadFinished(curr_level_size, true);
        std::vector<tree_node_t *> curr_level(curr_level_size);

        split_data_temp.resize(pow(2, i + 1));

#pragma omp parallel for
        for (int j = 0; j < curr_level_size; ++j)
        {
            int current_idx = current_layer_idx + j;
            thrust::device_vector<double> x_train_curr;
            thrust::device_vector<double> y_train_curr;

            x_train_curr = split_data_curr[j].first;
            y_train_curr = split_data_curr[j].second;

            int N = y_train_curr.size();

            double weight = 1.0 / N;
            double mean = thrust::reduce(thrust::cuda::par.on(streams[j % NUM_STREAMS]), y_train_curr.begin(), y_train_curr.end(), 0.0) / N;

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
            else if (i == depth || elements_equal_gpu(thrust::cuda::par.on(streams[j % NUM_STREAMS]), y_train_curr, N, THRESHOLD) ||
                     rows_equal_gpu(thrust::cuda::par.on(streams[j % NUM_STREAMS]), x_train_curr, D_global, N, THRESHOLD))
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

            split_output_t split = split_gpu(thrust::cuda::par.on(streams[j % NUM_STREAMS]), D_global, N, x_train_curr, y_train_curr);

            thrust::device_ptr<double> x_train_curr_ptr = x_train_curr.data();
            int D_local = D_global;

            int size_left = thrust::count_if(thrust::cuda::par.on(streams[j % NUM_STREAMS]),
                                             thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                                             [x_train_curr_ptr, split, D_local] __device__(int i)
                                             {
                                                 double x = x_train_curr_ptr[i * D_local + split.cut_feature];
                                                 return x <= split.cut_value;
                                             });
            int size_right = N - size_left;

            thrust::device_vector<int> left_indices(size_left);
            thrust::device_vector<int> right_indices(size_right);
            thrust::device_vector<int> left_x_indices(size_left * D_global);
            thrust::device_vector<int> right_x_indices(size_right * D_global);

            thrust::stable_partition_copy(thrust::cuda::par.on(streams[j % NUM_STREAMS]),
                                          thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                                          left_indices.begin(), right_indices.begin(),
                                          [x_train_curr_ptr, split, D_local] __device__(int i)
                                          {
                                              double x = x_train_curr_ptr[i * D_local + split.cut_feature];
                                              return x <= split.cut_value;
                                          });

            thrust::stable_partition_copy(thrust::cuda::par.on(streams[j % NUM_STREAMS]),
                                          thrust::make_counting_iterator(0), thrust::make_counting_iterator(N * D_global),
                                          left_x_indices.begin(), right_x_indices.begin(),
                                          [x_train_curr_ptr, split, D_local] __device__(int i)
                                          {
                                              double x = x_train_curr_ptr[(i / D_local) * D_local + split.cut_feature];
                                              return x <= split.cut_value;
                                          });

            thrust::device_vector<double> left_x_train(left_indices.size() * D_global);
            thrust::device_vector<double> right_x_train(right_indices.size() * D_global);
            thrust::device_vector<double> left_y_train(left_indices.size());
            thrust::device_vector<double> right_y_train(right_indices.size());

            thrust::gather(thrust::cuda::par.on(streams[j % NUM_STREAMS]), left_indices.begin(), left_indices.end(), y_train_curr.begin(), left_y_train.begin());
            thrust::gather(thrust::cuda::par.on(streams[j % NUM_STREAMS]), right_indices.begin(), right_indices.end(), y_train_curr.begin(), right_y_train.begin());
            thrust::gather(thrust::cuda::par.on(streams[j % NUM_STREAMS]), left_x_indices.begin(), left_x_indices.end(), x_train_curr.begin(), left_x_train.begin());
            thrust::gather(thrust::cuda::par.on(streams[j % NUM_STREAMS]), right_x_indices.begin(), right_x_indices.end(), x_train_curr.begin(), right_x_train.begin());

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

void compute_predictions(tree_node_t *tree)
{
    for (int i = 0; i < test_size_global; ++i)
    {
        std::vector<double> data = std::vector<double>(x_test_global.begin() + i * D_global, x_test_global.begin() + i * D_global + D_global);
        double prediction = eval_helper_gpu(tree, data);
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

struct RandomGenerator
{
    int a, b;

    __host__ __device__
    RandomGenerator(int _a, int _b) : a(_a), b(_b) {};

    __host__ __device__ float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_int_distribution<int> dist(a, b);
        rng.discard(n);

        return dist(rng);
    }
};

forest_t *build_forest(int depth, int num_trees)
{
    int subsample_size = ceil(SUBSAMPLE_RATE * train_size_global);
    forest_t *trees = new std::vector<tree_node_t *>(num_trees);
    thrust::device_vector<double> x_train_rand(subsample_size * D_global);
    thrust::device_vector<double> y_train_rand(subsample_size);

    thrust::device_vector<int> random_idx(subsample_size);
    tree_node_t *curr_tree;
    int D_local = D_global;

    thrust::device_ptr<double> x_train_global_ptr = (*x_train_global).data();
    thrust::device_ptr<double> x_train_rand_ptr = x_train_rand.data();
    thrust::device_ptr<int> random_idx_ptr = random_idx.data();

    for (int i = 0; i < num_trees; i++)
    {
        thrust::counting_iterator<unsigned int> index_sequence_begin(i * subsample_size);
        thrust::transform(thrust::device, index_sequence_begin,
                          index_sequence_begin + random_idx.size(),
                          random_idx.begin(),
                          RandomGenerator(0, train_size_global - 1));

        thrust::gather(thrust::device, random_idx.begin(), random_idx.end(),
                       (*y_train_global).begin(), y_train_rand.begin());

        for (int d = 0; d < D_global; d++)
        {
            thrust::counting_iterator<unsigned int> counting_iter(0);
            thrust::for_each(thrust::device,
                             counting_iter,
                             counting_iter + subsample_size,
                             [=] __device__(int idx)
                             {
                                 x_train_rand_ptr[idx * D_local + d] = x_train_global_ptr[random_idx_ptr[idx] * D_local + d];
                             });
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
            double prediction = eval_helper_gpu((*forest)[i], data);
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
