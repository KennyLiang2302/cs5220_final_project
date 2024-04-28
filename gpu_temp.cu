#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>


int round_down(int x, int D) {
    return x - (x % D);
} 

thrust::device_vector<int> argsort_exhaustive(thrust::device_vector<double>& x, int d, int D, int N) {
    thrust::device_vector<int> indices(D * N);
    thrust::sequence(indices.begin(), indices.end());

    thrust::sort(indices.begin(), indices.end(), [x] __device__ (int left_idx, int right_idx) {
        return x[round_down(left_idx, D) + d] < x[round_down(right_idx, D) + d];
    })
    return indices;
}

thrust::device_vector<int> argsort(thrust::device_vector<double>& x, int d, int D, int N) {
    thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end());

    thrust::sort(indices.begin(), indices.end(), [x] __device__ (int left_idx, int right_idx) {
        return x[left_idx*D + d] < x[right_idx*D + d];
    })
    return indices;
}

split_output_t split_serial(int D, int N, thrust::device_vector<double> &x_train, thrust::device_vector<double> &y_train)
{
    double weight = 1.0 / N;
    double min_loss = std::numeric_limits<double>::infinity();
    int feature = std::numeric_limits<int>::infinity();
    double cut_value = std::numeric_limits<double>::infinity();

    thrust::vector<split_output_t> split_results ((N-1) * D);

    // iterate through each feature
    for (int d = 0; d < D; ++d)
    {
        thrust::device_vector<int> indices_x = argsort_exhaustive(x_train, d, D, N);
        thrust::device_vector<int> indices_y = argsort(x_train, d, D, N);

        thrust::device_vector<double> x_train_sorted(N*D);
        thrust::device_vector<double> y_train_sorted(N);

        thrust::gather(thrust::device, indices_x.begin(), indices_x.end(), x_train.begin(), x_train_sorted.begin());
        thrust::gather(thrust::device, indices_y.begin(), indices_y.end(), y_train.begin(), y_train_sorted.begin());
        thrust::vector<double> y_train_sorted_squared(N);
        thrust::transform(y_train_sorted.begin(), y_train_sorted.end(), y_train_sorted_squared.begin(),
                       [] __device__ (double x) -> double
                       { return x * x; });

        double mean_square_left = 0.0;
        double mean_left = 0.0;
        double weight_left = 0.0;
        double mean_square_right = weight * thrust::reduce(y_train_sorted_squared.begin(), y_train_sorted_squared.end(), 0.0);
        double mean_right = weight * thrust::reduce(y_train_sorted.begin(), y_train_sorted.end(), 0.0);
        double weight_right = 1.0;

        thrust::device_vector<double> y_prefix_sum(N);
        thrust::device_vector<double> y_squared_prefix_sum(N);

        thrust::inclusive_scan(thrust::host, y_train_sorted.begin(), y_train_sorted.end(), y_prefix_sum.begin());
        thrust::inclusive_scan(thrust::host, y_train_sorted_squared.begin(), y_train_sorted_squared.begin(), y_squared_prefix_sum.begin());

        thrust::device_vector<int> split_indices(N-1);
        thrust::sequence(split_indices.begin(), split_indices.end());

        thrust::transform(split_indices.begin(), split_indices.end(), losses.begin(), 
                [N, d, split_results, weight, mean_square_right, mean_right, y_prefix_sum, y_squared_prefix_sum, x_train_sorted] __device__ (int index) {
                double mean_square_left = weight * y_squared_prefix_sum[index];
                double mean_left = weight * y_prefix_sum[index];
                double local_mean_right = mean_right - mean_left; 
                double local_mean_sq_right = mean_square_right - mean_square_left;
                double weight_left = (index + 1) * weight;
                double weight_right = (N - index - 1) * weight;
                double left_loss = mean_square_left - pow(mean_left, 2) / weight_left;
                double right_loss = local_mean_sq_right - pow(local_mean_right, 2) / weight_right;
                split_output_t split_output;
                split_output.cut_feature = d;
                split_output.cut_value = (x_train_sorted[index] + x_train_sorted[index + 1]) / 2;
                split_output.loss = left_loss + right_loss;
                split_results[(N-1)*d + index] = split_output;
            });

        cudaDeviceSynchronize();
    }

    split_output_t output = thrust::reduce(thrust::device, split_results.begin(), split_results.end(), split_results[0], 
            [] __device__ (split_output_t& left, split_output_t& right) {
            if (left.loss < right.loss) {
                return left;
            } else {
                return right;
            }
        });

    return output;
}
