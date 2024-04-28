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

        // Only consider splits with at least one value on each side
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
    return output;
}
