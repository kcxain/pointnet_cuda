#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)
/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    return params;
}

/****************************************************************************************
 * 读取测试集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
}

void debug_print_array(float* array, int M, int N, int m, int n) {
    size_t size = M * N;
    float* h_array = new float[size];
    cudaMemcpy(h_array, array, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        std::cout << h_array[m * N + i] << " ";
    }
    std::cout << std::endl;
}


// CUDA Kernels and functions
#define BLOCK_SIZE 16
#define EPSILON 1e-5

__global__ void conv1d_kernel(float* input, float* output, float* weight, float* bias, int N, int in_channels, int out_channels) {
    // input: (in_channels, N)
    // output: (out_channels, N)
    // weight: (out_channels, in_channels)
    // bias: (out_channels)
    int m = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    int n = blockIdx.x * blockDim.x + threadIdx.x; // 列索引

    if (m < out_channels && n < N) {
        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            sum += input[c * N + n] * weight[m * in_channels + c];
        }
        output[m * N + n] = sum + bias[m];
    }
}

void conv1d(float* input, float* output, float* weight, float* bias, int N, int in_channels, int out_channels) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv1d_kernel<<<gridDim, blockDim>>>(input, output, weight, bias, N, in_channels, out_channels);
}

__global__ void batchnorm1d_kernel(float* input, float* output, float* gamma, float* beta, float* running_mean, float* running_var, int channels, int N) {
    // input: (channels, N)
    // output: (channels, N)
    // gamma: (channels)
    // beta: (channels)
    // running_mean: (channels)
    // running_var: (channels)
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < channels && n < N) {
        int index = m * N + n;
        float mean = running_mean[m];
        float var = running_var[m];
        float x = input[index];
        float x_hat = (x - mean) / sqrtf(var + EPSILON);
        output[index] = gamma[m] * x_hat + beta[m];
    }
}

void batchnorm1d(float* input, float* output, float* gamma, float* beta, float* running_mean, float* running_var, int N, int channels) {
    // input: (channels, N)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((channels + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    batchnorm1d_kernel<<<gridDim, blockDim>>>(input, output, gamma, beta, running_mean, running_var, channels, N);
}

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

void relu(float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(input, output, size);
}

__global__ void max_pooling_kernel(float* input, float* output, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) {
        float max_val = -FLT_MAX;
        for (int n = 0; n < N; ++n) {
            float val = input[m * N + n];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[m] = max_val;
    }
}

void max_pooling(float* input, float* output, int M, int N) {
    // input: (M, N)
    // output: (M)
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    max_pooling_kernel<<<gridSize, blockSize>>>(input, output, M, N);
}

__global__ void linear_kernel(float* input, float* output, float* weight, float* bias, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[i] * weight[idx * in_features + i];
        }
        output[idx] = sum + bias[idx];
    }
}

void linear(float* input, float* output, float* weight, float* bias, int in_features, int out_features) {
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;
    linear_kernel<<<gridSize, blockSize>>>(input, output, weight, bias, in_features, out_features);
}

__global__ void log_softmax_kernel(float* input, float* output, int size) {
    float maxVal = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += expf(input[i] - maxVal);
    }
    float logSum = maxVal + logf(sum);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] - logSum;
    }
}

void log_softmax(float* input, float* output, int size) {
    log_softmax_kernel<<<1, 1>>>(input, output, size);
}

__global__ void matmul_kernel(float* A, float* B, float* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // N
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K

    if (row < N && col < K) {
        float val = 0.0f;
        for (int i = 0; i < M; ++i) {
            val += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = val;
    }
}

void matmul(float* A, float* B, float* C, int N, int M, int K) {
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, N, M, K);
}


__global__ void add_identity_kernel(float* input, float* output, int size) {
    // input: (size, size)
    // output: (size, size)
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if(m < size && n < size) {
        if (m == n) {
            output[m * size + n] = input[m * size + n] + 1;
        } else {
            output[m * size + n] = input[m * size + n];
        }
    }
}

void add_identity(float* input, float* output, int size) {
    // size: 方阵的边长
    dim3 blockSize(16, 16);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);
    add_identity_kernel<<<gridSize, blockSize>>>(input, output, size);
}

__global__ void transpose_kernel(float* input, float* output, int M, int N) {
    // input: (M, N)
    // output: (N, M)
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        output[n * M + m] = input[m * N + n];
    }
}

void transpose(float* input, float* output, int M, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<gridSize, blockSize>>>(input, output, M, N);
}

void free_cuda() {
    return;
}

int main(int argc, char *argv[]) {
    std::string dir = argv[1];  // 模型参数文件的目录

    // 读取模型参数
    auto params = read_params(dir);

    std::string file_path = "data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取测试集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    // 将参数复制到设备内存
    std::map<std::string, float*> d_params;
    for (auto& kv : params) {
        std::string param_name = kv.first;
        std::vector<float>& h_param_data = kv.second;

        float* d_param_data = nullptr;
        size_t size_in_bytes = h_param_data.size() * sizeof(float);

        CUDA_ERROR_CHECK(cudaMalloc(&d_param_data, size_in_bytes));
        CUDA_ERROR_CHECK(cudaMemcpy(d_param_data, h_param_data.data(), size_in_bytes, cudaMemcpyHostToDevice));

        d_params[param_name] = d_param_data;
    }

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;
    int total = list_of_points.size();

    for (size_t i = 0; i < list_of_points.size(); i++) {
    // for (size_t i = 0; i < 1; i++) {
        auto &points = list_of_points[i]; // size [N * 3]
        int N = points.size() / 3; // N个点
        // for (int j = 0; j < 10; j++) {
        //     printf("%f ", points[j]);
        // }
        // return;
        // 将points复制到设备内存
        float* d_points = nullptr;
        CUDA_ERROR_CHECK(cudaMalloc(&d_points, points.size() * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMemcpy(d_points, points.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice));

        // debug_print_array(d_points, N, 3, 20, 3);
        // points : (N,3)

        // points_T : (3,N)
        float* d_points_T;
        CUDA_ERROR_CHECK(cudaMalloc(&d_points_T, points.size() * sizeof(float)));
        transpose(d_points, d_points_T, N, 3);
        // debug_print_array(d_points_T, 3, N, 1, 50);
        // PASS

        // =========STN3d=========
        // feat_stn_conv1_res = conv1d(points.T,"feat.stn.conv1.weight","feat.stn.conv1.bias",3,64)
        // (3, N) -> (64, N)
        float* d_feat_stn_conv1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_conv1_out, N * 64 * sizeof(float)));
        conv1d(d_points_T, d_feat_stn_conv1_out, d_params["feat.stn.conv1.weight"], d_params["feat.stn.conv1.bias"], N, 3, 64);
        // debug_print_array(d_feat_stn_conv1_out, 64, N, 1, 10);
        // PASS

        // feat_stn_bn1_res = bacthnorm1d(feat_stn_conv1_res,"feat.stn.bn1.weight","feat.stn.bn1.bias","feat.stn.bn1.running_mean","feat.stn.bn1.running_var",64)
        float* d_feat_stn_bn1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_bn1_out, N * 64 * sizeof(float)));
        batchnorm1d(d_feat_stn_conv1_out, d_feat_stn_bn1_out, d_params["feat.stn.bn1.weight"], d_params["feat.stn.bn1.bias"], d_params["feat.stn.bn1.running_mean"], d_params["feat.stn.bn1.running_var"], N, 64);
        // debug_print_array(d_feat_stn_bn1_out, 64, N, 10, 50);
        // PASS

        // feat_stn_relu_bn1_res = np.where(feat_stn_bn1_res>0,feat_stn_bn1_res,0)
        float* d_feat_stn_relu_bn1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_relu_bn1_out, N * 64 * sizeof(float)));
        relu(d_feat_stn_bn1_out, d_feat_stn_relu_bn1_out, N * 64);
        // debug_print_array(d_feat_stn_relu_bn1_out, 64, N, 10, 50);
        // PASS

        // =================================
        // feat_stn_conv2
        float* d_feat_stn_conv2_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_conv2_out, N * 128 * sizeof(float)));
        conv1d(d_feat_stn_relu_bn1_out, d_feat_stn_conv2_out, d_params["feat.stn.conv2.weight"], d_params["feat.stn.conv2.bias"], N, 64, 128);

        // feat_stn_bn2
        float* d_feat_stn_bn2_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_bn2_out, N * 128 * sizeof(float)));
        batchnorm1d(d_feat_stn_conv2_out, d_feat_stn_bn2_out, d_params["feat.stn.bn2.weight"], d_params["feat.stn.bn2.bias"], d_params["feat.stn.bn2.running_mean"], d_params["feat.stn.bn2.running_var"], N, 128);

        // relu
        float* d_feat_stn_relu_bn2_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_relu_bn2_out, N * 128 * sizeof(float)));
        relu(d_feat_stn_bn2_out, d_feat_stn_relu_bn2_out, N * 128);
        // debug_print_array(d_feat_stn_relu_bn2_out, 128, N, 10, 50);
        // PASS

        // =================================
        // feat_stn_conv3
        float* d_feat_stn_conv3_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_conv3_out, N * 1024 * sizeof(float)));
        conv1d(d_feat_stn_relu_bn2_out, d_feat_stn_conv3_out, d_params["feat.stn.conv3.weight"], d_params["feat.stn.conv3.bias"], N, 128, 1024);

        // feat_stn_bn3
        float* d_feat_stn_bn3_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_bn3_out, N * 1024 * sizeof(float)));
        batchnorm1d(d_feat_stn_conv3_out, d_feat_stn_bn3_out, d_params["feat.stn.bn3.weight"], d_params["feat.stn.bn3.bias"], d_params["feat.stn.bn3.running_mean"], d_params["feat.stn.bn3.running_var"], N, 1024);

        // relu
        float* d_feat_stn_relu_bn3_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_relu_bn3_out, N * 1024 * sizeof(float)));
        relu(d_feat_stn_bn3_out, d_feat_stn_relu_bn3_out, N * 1024);
        // debug_print_array(d_feat_stn_relu_bn3_out, 1024, N, 10, 50);
        // PASS

        // max pooling over N
        float* d_feat_stn_maxpool_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_maxpool_out, 1024 * sizeof(float)));
        max_pooling(d_feat_stn_relu_bn3_out, d_feat_stn_maxpool_out, 1024, N);
        // debug_print_array(d_feat_stn_maxpool_out, 1, 1024, 0, 50);
        // PASS
        // =================================

        // feat_stn_fc1_res
        float* d_feat_stn_fc1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_fc1_out, 512 * sizeof(float)));
        linear(d_feat_stn_maxpool_out, d_feat_stn_fc1_out, d_params["feat.stn.fc1.weight"], d_params["feat.stn.fc1.bias"], 1024, 512);
        // DEBUG: Linear 操作后，差值较大
        //debug_print_array(d_feat_stn_fc1_out, 1, 512, 0, 50);
        // PASS

        // feat_stn_bn4_res
        float* d_feat_stn_bn4_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_bn4_out, 512 * sizeof(float)));
        batchnorm1d(d_feat_stn_fc1_out, d_feat_stn_bn4_out, d_params["feat.stn.bn4.weight"], d_params["feat.stn.bn4.bias"], d_params["feat.stn.bn4.running_mean"], d_params["feat.stn.bn4.running_var"], 1, 512);

        // feat_stn_relu_bn4_res
        float* d_feat_stn_relu_bn4_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_relu_bn4_out, 512 * sizeof(float)));
        relu(d_feat_stn_bn4_out, d_feat_stn_relu_bn4_out, 512);

        // feat_stn_fc2_res
        float* d_feat_stn_fc2_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_fc2_out, 256 * sizeof(float)));
        linear(d_feat_stn_relu_bn4_out, d_feat_stn_fc2_out, d_params["feat.stn.fc2.weight"], d_params["feat.stn.fc2.bias"], 512, 256);

        // feat_stn_bn5_res
        float* d_feat_stn_bn5_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_bn5_out, 256 * sizeof(float)));
        batchnorm1d(d_feat_stn_fc2_out, d_feat_stn_bn5_out, d_params["feat.stn.bn5.weight"], d_params["feat.stn.bn5.bias"], d_params["feat.stn.bn5.running_mean"], d_params["feat.stn.bn5.running_var"], 1, 256);

        // feat_stn_relu_bn5_res
        float* d_feat_stn_relu_bn5_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_relu_bn5_out, 256 * sizeof(float)));
        relu(d_feat_stn_bn5_out, d_feat_stn_relu_bn5_out, 256);
        // debug_print_array(d_feat_stn_relu_bn5_out, 1, 256, 0, 50);
        // PASS

        // feat_stn_fc3_res
        float* d_feat_stn_fc3_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_fc3_out, 9 * sizeof(float)));
        linear(d_feat_stn_relu_bn5_out, d_feat_stn_fc3_out, d_params["feat.stn.fc3.weight"], d_params["feat.stn.fc3.bias"], 256, 9);
        // debug_print_array(d_feat_stn_fc3_out, 1, 9, 0, 9);
        // PASS

        // feat_stn_res
        float* d_feat_stn_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_stn_res, 9 * sizeof(float)));
        add_identity(d_feat_stn_fc3_out, d_feat_stn_res, 3);
        //debug_print_array(d_feat_stn_res, 1, 9, 0, 9);
        // PASS
        // =========end STN3d=========
        

        float* d_transformed_points;
        CUDA_ERROR_CHECK(cudaMalloc(&d_transformed_points, N * 3 * sizeof(float)));

        matmul(d_points, d_feat_stn_res, d_transformed_points, N, 3, 3); // [N,3] * [3,3] = [N,3]
        // debug_print_array(d_transformed_points, N, 3, 10, 3);
        // PASS

        float* d_transformed_points_T;
        CUDA_ERROR_CHECK(cudaMalloc(&d_transformed_points_T, N * 3 * sizeof(float)));
        transpose(d_transformed_points, d_transformed_points_T, N, 3);

        // feat_conv1
        float* d_feat_conv1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_conv1_out, N * 64 * sizeof(float)));
        conv1d(d_transformed_points_T, d_feat_conv1_out, d_params["feat.conv1.weight"], d_params["feat.conv1.bias"], N, 3, 64);
        // debug_print_array(d_feat_conv1_out, 64, N, 10, 50);
        // PASS

        // feat_bn1
        float* d_feat_bn1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_bn1_out, N * 64 * sizeof(float)));
        batchnorm1d(d_feat_conv1_out, d_feat_bn1_out, d_params["feat.bn1.weight"], d_params["feat.bn1.bias"], d_params["feat.bn1.running_mean"], d_params["feat.bn1.running_var"], N, 64);

        // feat_relu_bn1
        float* d_feat_relu_bn1_out;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_relu_bn1_out, N * 64 * sizeof(float)));
        relu(d_feat_bn1_out, d_feat_relu_bn1_out, N * 64);
        // debug_print_array(d_feat_relu_bn1_out, 64, N, 60, 50);
        // PASS

        // =========STNkd=========

        // feat_fstn_conv1_res
        float* d_feat_fstn_conv1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_conv1_res, N * 64 * sizeof(float)));
        conv1d(d_feat_relu_bn1_out, d_feat_fstn_conv1_res, d_params["feat.fstn.conv1.weight"], d_params["feat.fstn.conv1.bias"], N, 64, 64);

        // feat_fstn_bn1_res
        float* d_feat_fstn_bn1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_bn1_res, N * 64 * sizeof(float)));
        batchnorm1d(d_feat_fstn_conv1_res, d_feat_fstn_bn1_res, d_params["feat.fstn.bn1.weight"], d_params["feat.fstn.bn1.bias"], d_params["feat.fstn.bn1.running_mean"], d_params["feat.fstn.bn1.running_var"], N, 64);

        // feat_fstn_relu_bn1_res
        float* d_feat_fstn_relu_bn1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_relu_bn1_res, N * 64 * sizeof(float)));
        relu(d_feat_fstn_bn1_res, d_feat_fstn_relu_bn1_res, N * 64);
        // debug_print_array(d_feat_fstn_relu_bn1_res, 64, N, 30, 50);
        // PASS

        // feat_fstn_conv2_res
        float* d_feat_fstn_conv2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_conv2_res, N * 128 * sizeof(float)));
        conv1d(d_feat_fstn_relu_bn1_res, d_feat_fstn_conv2_res, d_params["feat.fstn.conv2.weight"], d_params["feat.fstn.conv2.bias"], N, 64, 128);

        // feat_fstn_bn2_res
        float* d_feat_fstn_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_bn2_res, N * 128 * sizeof(float)));
        batchnorm1d(d_feat_fstn_conv2_res, d_feat_fstn_bn2_res, d_params["feat.fstn.bn2.weight"], d_params["feat.fstn.bn2.bias"], d_params["feat.fstn.bn2.running_mean"], d_params["feat.fstn.bn2.running_var"], N, 128);

        // relu
        float* d_feat_fstn_relu_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_relu_bn2_res, N * 128 * sizeof(float)));
        relu(d_feat_fstn_bn2_res, d_feat_fstn_relu_bn2_res, N * 128);

        // feat_fstn_conv3_res
        float* d_feat_fstn_conv3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_conv3_res, N * 1024 * sizeof(float)));
        conv1d(d_feat_fstn_relu_bn2_res, d_feat_fstn_conv3_res, d_params["feat.fstn.conv3.weight"], d_params["feat.fstn.conv3.bias"], N, 128, 1024);

        // feat_fstn_bn3_res
        float* d_feat_fstn_bn3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_bn3_res, N * 1024 * sizeof(float)));
        batchnorm1d(d_feat_fstn_conv3_res, d_feat_fstn_bn3_res, d_params["feat.fstn.bn3.weight"], d_params["feat.fstn.bn3.bias"], d_params["feat.fstn.bn3.running_mean"], d_params["feat.fstn.bn3.running_var"], N, 1024);

        // relu
        float* d_feat_fstn_relu_bn3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_relu_bn3_res, N * 1024 * sizeof(float)));
        relu(d_feat_fstn_bn3_res, d_feat_fstn_relu_bn3_res, N * 1024);

        //feat_fstn_max_res
        // max pooling over N
        float* d_feat_fstn_max_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_max_res, 1024 * sizeof(float)));
        max_pooling(d_feat_fstn_relu_bn3_res, d_feat_fstn_max_res, 1024, N);
        // debug_print_array(d_feat_fstn_max_res, 1, 1024, 0, 50);
        // PASS

        // feat_fstn_fc1_res
        float* d_feat_fstn_fc1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_fc1_res, 512 * sizeof(float)));
        linear(d_feat_fstn_max_res, d_feat_fstn_fc1_res, d_params["feat.fstn.fc1.weight"], d_params["feat.fstn.fc1.bias"], 1024, 512);

        // feat_fstn_bn4_res
        float* d_feat_fstn_bn4_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_bn4_res, 512 * sizeof(float)));
        batchnorm1d(d_feat_fstn_fc1_res, d_feat_fstn_bn4_res, d_params["feat.fstn.bn4.weight"], d_params["feat.fstn.bn4.bias"], d_params["feat.fstn.bn4.running_mean"], d_params["feat.fstn.bn4.running_var"], 1, 512);

        // relu
        float* d_feat_fstn_relu_bn4_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_relu_bn4_res, 512 * sizeof(float)));
        relu(d_feat_fstn_bn4_res, d_feat_fstn_relu_bn4_res, 512);

        // feat_fstn_fc2_res
        float* d_feat_fstn_fc2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_fc2_res, 256 * sizeof(float)));
        linear(d_feat_fstn_relu_bn4_res, d_feat_fstn_fc2_res, d_params["feat.fstn.fc2.weight"], d_params["feat.fstn.fc2.bias"], 512, 256);

        // feat_fstn_bn5_res
        float* d_feat_fstn_bn5_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_bn5_res, 256 * sizeof(float)));
        batchnorm1d(d_feat_fstn_fc2_res, d_feat_fstn_bn5_res, d_params["feat.fstn.bn5.weight"], d_params["feat.fstn.bn5.bias"], d_params["feat.fstn.bn5.running_mean"], d_params["feat.fstn.bn5.running_var"], 1, 256);

        // relu
        float* d_feat_fstn_relu_bn5_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_relu_bn5_res, 256 * sizeof(float)));
        relu(d_feat_fstn_bn5_res, d_feat_fstn_relu_bn5_res, 256);

        // feat_fstn_fc3_res
        float* d_feat_fstn_fc3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_fc3_res, 64 * 64 * sizeof(float)));
        linear(d_feat_fstn_relu_bn5_res, d_feat_fstn_fc3_res, d_params["feat.fstn.fc3.weight"], d_params["feat.fstn.fc3.bias"], 256, 64 * 64);
        // debug_print_array(d_feat_fstn_fc3_res, 1, 64 * 64, 0, 50);
        // PASS

        float* d_feat_fstn_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_fstn_res, 64 * 64 * sizeof(float)));
        add_identity(d_feat_fstn_fc3_res, d_feat_fstn_res, 64);
        //debug_print_array(d_feat_fstn_res, 1, 64 * 64, 0, 60);
        // PASS
        // =========end STNkd=========

        float* d_trans_feat = d_feat_fstn_res;
        // 将feat_relu_bn1_res转置
        float* d_feat_relu_bn1_res_T;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_relu_bn1_res_T, N * 64 * sizeof(float)));
        transpose(d_feat_relu_bn1_out, d_feat_relu_bn1_res_T, 64, N);

        // feat_bmm_res2 = feat_relu_bn1_res.T @ trans_feat
        float* d_feat_bmm_res2;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_bmm_res2, N * 64 * sizeof(float)));
        matmul(d_feat_relu_bn1_res_T, d_trans_feat, d_feat_bmm_res2, N, 64, 64);

        // 将feat_bmm_res2转置回去
        float* d_feat_bmm_res2_T;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_bmm_res2_T, N * 64 * sizeof(float)));
        transpose(d_feat_bmm_res2, d_feat_bmm_res2_T, N, 64);

        // feat_conv2_res = conv1d(feat_bmm_res2_T, "feat.conv2.weight", "feat.conv2.bias", 64, 128)
        float* d_feat_conv2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_conv2_res, N * 128 * sizeof(float)));
        conv1d(d_feat_bmm_res2_T, d_feat_conv2_res, d_params["feat.conv2.weight"], d_params["feat.conv2.bias"], N, 64, 128);

        // feat_bn2_res
        float* d_feat_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_bn2_res, N * 128 * sizeof(float)));
        batchnorm1d(d_feat_conv2_res, d_feat_bn2_res, d_params["feat.bn2.weight"], d_params["feat.bn2.bias"], d_params["feat.bn2.running_mean"], d_params["feat.bn2.running_var"], N, 128);

        // relu
        float* d_feat_relu_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_relu_bn2_res, N * 128 * sizeof(float)));
        relu(d_feat_bn2_res, d_feat_relu_bn2_res, N * 128);

        // feat_conv3_res = conv1d(feat_relu_bn2_res, "feat.conv3.weight", "feat.conv3.bias", 128, 1024)
        float* d_feat_conv3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_conv3_res, N * 1024 * sizeof(float)));
        conv1d(d_feat_relu_bn2_res, d_feat_conv3_res, d_params["feat.conv3.weight"], d_params["feat.conv3.bias"], N, 128, 1024);

        // feat_bn3_res
        float* d_feat_bn3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_bn3_res, N * 1024 * sizeof(float)));
        batchnorm1d(d_feat_conv3_res, d_feat_bn3_res, d_params["feat.bn3.weight"], d_params["feat.bn3.bias"], d_params["feat.bn3.running_mean"], d_params["feat.bn3.running_var"], N, 1024);

        // feat_max_res = np.max(feat_bn3_res, axis=1)
        float* d_feat_max_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_feat_max_res, 1024 * sizeof(float)));
        max_pooling(d_feat_bn3_res, d_feat_max_res, 1024, N);
        // debug_print_array(d_feat_max_res, 1, 1024, 0, 50);
        // PASS
        // ============end PointNetEncoder============

        // fc1_res = linear(feat_max_res, "fc1.weight", "fc1.bias", 1024, 512)
        float* d_fc1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_fc1_res, 512 * sizeof(float)));
        linear(d_feat_max_res, d_fc1_res, d_params["fc1.weight"], d_params["fc1.bias"], 1024, 512);

        // bn1_res
        float* d_bn1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_bn1_res, 512 * sizeof(float)));
        batchnorm1d(d_fc1_res, d_bn1_res, d_params["bn1.weight"], d_params["bn1.bias"], d_params["bn1.running_mean"], d_params["bn1.running_var"], 1, 512);

        // relu
        float* d_relu_bn1_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_relu_bn1_res, 512 * sizeof(float)));
        relu(d_bn1_res, d_relu_bn1_res, 512);

        // fc2_res = linear(relu_bn1_res, "fc2.weight", "fc2.bias", 512, 256)
        float* d_fc2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_fc2_res, 256 * sizeof(float)));
        linear(d_relu_bn1_res, d_fc2_res, d_params["fc2.weight"], d_params["fc2.bias"], 512, 256);

        // bn2_res
        float* d_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_bn2_res, 256 * sizeof(float)));
        batchnorm1d(d_fc2_res, d_bn2_res, d_params["bn2.weight"], d_params["bn2.bias"], d_params["bn2.running_mean"], d_params["bn2.running_var"], 1, 256);

        // relu
        float* d_relu_bn2_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_relu_bn2_res, 256 * sizeof(float)));
        relu(d_bn2_res, d_relu_bn2_res, 256);

        // fc3_res = linear(relu_bn2_res, "fc3.weight", "fc3.bias", 256, 10)
        float* d_fc3_res;
        CUDA_ERROR_CHECK(cudaMalloc(&d_fc3_res, 10 * sizeof(float)));
        linear(d_relu_bn2_res, d_fc3_res, d_params["fc3.weight"], d_params["fc3.bias"], 256, 10);

        // 推理不需要 softmax
        // softmax_res = log_softmax(fc3_res)
        // float* d_softmax_res;
        // CUDA_ERROR_CHECK(cudaMalloc(&d_softmax_res, 10 * sizeof(float)));
        // log_softmax(d_fc3_res, d_softmax_res, 10);
        // debug_print_array(d_softmax_res, 1, 10, 0, 10);
        // PASS

        // 从设备复制输出到主机
        float h_output[10];
        CUDA_ERROR_CHECK(cudaMemcpy(h_output, d_fc3_res, 10 * sizeof(float), cudaMemcpyDeviceToHost));

        // 计算预测结果
        int predicted_label = std::distance(h_output, std::max_element(h_output, h_output + 10));
        // printf("%d\n", predicted_label);

        if (predicted_label == list_of_labels[i]) {
            correct++;
        }
        CUDA_ERROR_CHECK(cudaFree(d_points));
        CUDA_ERROR_CHECK(cudaFree(d_points_T));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_conv1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_bn1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_relu_bn1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_conv2_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_bn2_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_relu_bn2_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_conv3_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_bn3_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_relu_bn3_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_maxpool_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_fc1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_bn4_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_relu_bn4_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_fc2_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_bn5_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_relu_bn5_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_fc3_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_stn_res));
        CUDA_ERROR_CHECK(cudaFree(d_transformed_points));
        CUDA_ERROR_CHECK(cudaFree(d_transformed_points_T));
        CUDA_ERROR_CHECK(cudaFree(d_feat_conv1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_bn1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_relu_bn1_out));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_conv1_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_bn1_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_relu_bn1_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_conv2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_relu_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_conv3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_bn3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_relu_bn3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_max_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_fc1_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_bn4_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_relu_bn4_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_fc2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_bn5_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_relu_bn5_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_fc3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_fstn_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_relu_bn1_res_T));
        CUDA_ERROR_CHECK(cudaFree(d_feat_bmm_res2));
        CUDA_ERROR_CHECK(cudaFree(d_feat_bmm_res2_T));
        CUDA_ERROR_CHECK(cudaFree(d_feat_conv2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_relu_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_conv3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_bn3_res));
        CUDA_ERROR_CHECK(cudaFree(d_feat_max_res));
        CUDA_ERROR_CHECK(cudaFree(d_fc1_res));
        CUDA_ERROR_CHECK(cudaFree(d_bn1_res));
        CUDA_ERROR_CHECK(cudaFree(d_relu_bn1_res));
        CUDA_ERROR_CHECK(cudaFree(d_fc2_res));
        CUDA_ERROR_CHECK(cudaFree(d_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_relu_bn2_res));
        CUDA_ERROR_CHECK(cudaFree(d_fc3_res));
        // CUDA_ERROR_CHECK(cudaFree(d_softmax_res));
    }

    // 同步设备
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 计算准确率
    double accuracy = static_cast<double>(correct) / total;

    // 输出结果
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy;

    // 释放参数的设备内存
    for (auto& kv : d_params) {
        cudaFree(kv.second);
    }

    return 0;
}