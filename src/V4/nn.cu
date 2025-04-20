#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define INPUT_SIZE 784 // 28 x 28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 32
#define NUM_CLASSES 10  // Digits 0-9
#define train_amount 60000
#define test_amount 10000
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
float* allocateMatrix(int rows, int cols) {
    return (float*)malloc(rows * cols * sizeof(float));
}

// Activation functions
__device__ float relu(float x) {
    if(x > 0)
        return x;
    else
        return 0;
}

// Neural network structure
typedef struct {
    float* W1;
    float* W2;
    float* b1;
    float* b2;
} NeuralNetwork;

typedef struct { 
    NeuralNetwork* host_interface;
    NeuralNetwork* device_interface;
} NNInterface;

// Initialize neural network
NNInterface* createNetwork() {
    // Step 1
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    float* W1_h = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    float* W2_h = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    float* b1_h = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* b2_h = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    srand(1234); // Kept the same for testing
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            W1_h[i * INPUT_SIZE + j] = ((float)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W2_h[i * HIDDEN_SIZE + j] = ((float)rand() / RAND_MAX) * 0.01;

    // Step 2
    float* W1_d;
    float* W2_d;
    float* b1_d;
    float* b2_d;

    cudaMalloc((void**)&W1_d, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&W2_d, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&b1_d, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&b2_d, OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(W1_d, W1_h, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2_d, W2_h, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice); // These aren't really required but
    cudaMemcpy(b2_d, b2_h, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice); // better safe than sorry. calloc is used for a reason.

    // Step 3
    net->W1 = W1_d;
    net->W2 = W2_d;
    net->b1 = b1_d;
    net->b2 = b2_d;

    // Step 4
    NeuralNetwork* net_d;
    cudaMalloc((void**)&net_d, sizeof(NeuralNetwork));
    cudaMemcpy(net_d, net, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);

    free(W1_h);
    free(W2_h);
    free(b1_h);
    free(b2_h);

    NNInterface* nni = (NNInterface*)calloc(1, sizeof(NNInterface));
    nni->host_interface = net;
    nni->device_interface = net_d;

    return nni;
}
__device__ void matmul_tensor_cores(__half* A, __half* B, float* result, int r_row, int r_col, int A_col)
{
    int warpM  = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN  = (blockIdx.y * blockDim.y + threadIdx.y);
    if ((warpM * WMMA_M >= r_row) || (warpN * WMMA_N >= r_col)) 
        return;
    //declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);  //initialize output to zero values..
    //looping over K dimension
    for (int i = 0; i < A_col; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        nvcuda::wmma::load_matrix_sync(a_frag, &A[aRow * A_col + aCol], A_col);
        nvcuda::wmma::load_matrix_sync(b_frag, &B[bRow * r_col + bCol], r_col);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    //storing the results
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&result[cRow * r_col + cCol], c_frag, r_col, nvcuda::wmma::mem_row_major);
}


// Forward pass
__global__ void forward(NeuralNetwork* net, float* input,  float* hidden , __half* input_half, float* output, __half* W1_half) {
    //adding code to utilize tensor coress/////////////////////////////////////////////]
    // <<<BATCH_SIZE, 128>>>
    int b = blockIdx.x;
    int i = threadIdx.x;
    __shared__ float sum;
    // Compute hidden layer (input × W1 + b1)
    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE) {
        hidden[b * HIDDEN_SIZE + i] = net->b1[i];
        matmul_tensor_cores(input_half, W1_half, hidden, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
       // for (int j = 0; j < INPUT_SIZE; j++) {
       //     hidden[b * HIDDEN_SIZE + i] += net->W1[i * INPUT_SIZE + j] * input[b * INPUT_SIZE + j];
       // }
    }
    // Apply ReLU activation
    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE) {
        if (hidden[b * HIDDEN_SIZE + i] < 0) {
            hidden[b * HIDDEN_SIZE + i] = 0.0;
        }
    }
    // Compute output layer (hidden × W2 + b2)
    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to OUTPUT_SIZE (10)
    if(i < OUTPUT_SIZE) {
        output[b * OUTPUT_SIZE + i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[b * OUTPUT_SIZE + i] += net->W2[i * HIDDEN_SIZE + j] * hidden[b * HIDDEN_SIZE + j];
        }
        output[b * OUTPUT_SIZE + i] = exp(output[b * OUTPUT_SIZE + i]);
    }
    // Softmax
    if(i >= OUTPUT_SIZE) {
        return;
    }
    if(i == 0) { // atomicAdd for float's wasn't working. One thread finds the sum.
        sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += output[b * OUTPUT_SIZE + j];
        }
        // printf("sum: %f, block: %i\n", sum, b);
    }
    __syncthreads();
    // Loop: i = 0 to OUTPUT_SIZE
    output[b * OUTPUT_SIZE + i] /= sum;
}

// Backpropagation
__global__ void backward(NeuralNetwork* net, float* input, float* hidden, float* output, float* target, float* gradient_W1, float* gradient_W2, float* gradient_b1, float* gradient_b2, float* d_output, float* d_hidden) {

    int b = blockIdx.x;
    int i = threadIdx.x;

    // Compute d_output = output - target

    // Loop 1: b to BATCH_SIZE (Multiple of 32)
    // Loop 2: i to OUTPUT_SIZE (10)
    if(i < OUTPUT_SIZE) {
        d_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] - target[b * OUTPUT_SIZE + i];
    }
    __syncthreads();
    
    // Compute d_hidden

    // Loop 1: b to BATCH_SIZE (Multiple of 32)
    // Loop 2: i to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE) {
        float sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            sum += net->W2[j * HIDDEN_SIZE + i] * d_output[b * OUTPUT_SIZE + j];
    
        d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0) ? sum : 0.0;
    }
}

__global__ void update_grads_1(float* gradient_W1, float* gradient_W2, float* gradient_b1, float* gradient_b2, float* d_hidden, float* d_output, float* input, float* hidden) {

    // Accumulate gradients for W2 and b2

    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to OUTPUT_SIZE (10)
    // Loop 3: j = 0 to HIDDEN_SIZE (128)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < HIDDEN_SIZE * OUTPUT_SIZE) {
        int i = idx / HIDDEN_SIZE; // 0 to OUTPUT_SIZE
        int j = idx % HIDDEN_SIZE; // 0 to HIDDEN_SIZE
        for(int b = 0; b < BATCH_SIZE; b++) {
            gradient_W2[i * HIDDEN_SIZE + j] += d_output[b * OUTPUT_SIZE + i] * hidden[b * HIDDEN_SIZE + j];
        }
    }

    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to OUTPUT_SIZE (10)
    if(idx < OUTPUT_SIZE) {
        int i = idx; // 0 to OUTPUT_SIZE
        for(int b = 0; b < BATCH_SIZE; b++) {
            gradient_b2[i] += d_output[b * OUTPUT_SIZE + i];
        }
    }

    // Accumulate gradients for W1 and b1
    
    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to HIDDEN_SIZE (128)
    // Loop 3: j = 0 to INPUT_SIZE (784)
    if(idx < INPUT_SIZE * HIDDEN_SIZE) {
        int i = idx / INPUT_SIZE; // 0 to HIDDEN_SIZE
        int j = idx % INPUT_SIZE; // 0 to INPUT_SIZE
        for(int b = 0; b < BATCH_SIZE; b++) {
            gradient_W1[i * INPUT_SIZE + j] += d_hidden[b * HIDDEN_SIZE + i] * input[b * INPUT_SIZE + j];
        }
    }

    // Loop 1: b to BATCH_SIZE (Multiple of 32)
    // Loop 2: i to HIDDEN_SIZE (128)
    if(idx < HIDDEN_SIZE) {
        int i = idx; // 0 to HIDDEN_SIZE
        for(int b = 0; b < BATCH_SIZE; b++) {
            gradient_b1[i] += d_hidden[b * HIDDEN_SIZE + i];
        }
    }
}

__global__ void update_grads_2(float* gradient_W1, float* gradient_W2, float* gradient_b1, float* gradient_b2) {
    // Average gradients
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop: i to HIDDEN_SIZE * INPUT_SIZE (128 * 784 = 100352)
    if(i < HIDDEN_SIZE * INPUT_SIZE)
        gradient_W1[i] /= BATCH_SIZE;

    // Loop: i to OUTPUT_SIZE * HIDDEN_SIZE (10 * 128 = 1280)
    if(i < OUTPUT_SIZE * HIDDEN_SIZE)
        gradient_W2[i] /= BATCH_SIZE;

    // Loop: i to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE)
        gradient_b1[i] /= BATCH_SIZE;

    if(i < OUTPUT_SIZE)
        gradient_b2[i] /= BATCH_SIZE;
}

__global__ void update_net(NeuralNetwork* net, float* gradient_W1, float* gradient_W2, float* gradient_b1, float* gradient_b2) {
    // Update weights and biases
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Loop: i to OUTPUT_SIZE * HIDDEN_SIZE (10 * 128 = 1280)
    if(i < OUTPUT_SIZE * HIDDEN_SIZE)
        net->W2[i] -= LEARNING_RATE * gradient_W2[i];
    // Loop: i to HIDDEN_SIZE * INPUT_SIZE (128 * 784 = 100352)
    if(i < HIDDEN_SIZE * INPUT_SIZE)
        net->W1[i] -= LEARNING_RATE * gradient_W1[i];
    // Loop: i to OUTPUT_SIZE (10)
    if(i < OUTPUT_SIZE)
        net->b2[i] -= LEARNING_RATE * gradient_b2[i];
    // Loop: i to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE)
        net->b1[i] -= LEARNING_RATE * gradient_b1[i];
}
////////////////////////
__global__ void convertToHalf(float* input, __half* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = __float2half(input[i]);
    }
}

//////////////////////
// Train network
void train(NNInterface* nni, float* images, float* labels, int numImages, int numLabels) {

    float* images_d;
    cudaMalloc((void**)&images_d, numImages * INPUT_SIZE * sizeof(float));
    cudaMemcpy(images_d, images, numImages * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float* labels_d;
    cudaMalloc((void**)&labels_d, numLabels * OUTPUT_SIZE * sizeof(float));
    cudaMemcpy(labels_d, labels, numLabels * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    /////////////////////////////////////
    int size_1 = BATCH_SIZE * HIDDEN_SIZE;
    int size_2 = HIDDEN_SIZE * INPUT_SIZE;
    int threads__2 = 256;
    int blocks_1 = (size_1 + threads__2 - 1) / threads__2;
    int blocks_2 = (size_2 + threads__2 - 1) / threads__2;

    ////////////////////////////////////
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            if(i + BATCH_SIZE >= numImages) { // Safety to adjust for total images and batch sizes if it doesn't equally divide
                i = numImages - BATCH_SIZE;
            }
            __half* hidden_d_half;
            __half* W1_half;
            float* hidden_d;
            float* output_d;
            cudaMalloc((void**)&hidden_d, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
            cudaMalloc((void**)&output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            cudaMalloc((void**)&hidden_d_half, BATCH_SIZE * HIDDEN_SIZE * sizeof(__half));
            cudaMalloc((void**)&W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
            float* d_hidden;
            float* d_output;
            cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
            cudaMalloc((void**)&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

            float* sum;
            cudaMalloc((void**)&sum, sizeof(float));
    
            int max_i_in_forward = 128;


            convertToHalf<<<blocks_1, threads__2>>>((float*)hidden_d, hidden_d_half, size_1);
            convertToHalf<<<blocks_2, threads__2>>>((float*)nni->device_interface->W1, W1_half, size_2);

            forward<<<BATCH_SIZE, max_i_in_forward>>>(nni->device_interface, images_d + (i * INPUT_SIZE), hidden_d, hidden_d_half, output_d, W1_half);
            cudaDeviceSynchronize();

            float* gradient_W1;
            float* gradient_W2;
            float* gradient_b1;
            float* gradient_b2;

            cudaMalloc((void**)&gradient_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            cudaMalloc((void**)&gradient_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
            cudaMalloc((void**)&gradient_b1, HIDDEN_SIZE * sizeof(float));
            cudaMalloc((void**)&gradient_b2, OUTPUT_SIZE * sizeof(float));

            int max_i_in_backward = 128;
            backward<<<BATCH_SIZE, max_i_in_backward>>>(nni->device_interface, images_d + (i * INPUT_SIZE), hidden_d, output_d, labels_d + (i * OUTPUT_SIZE), gradient_W1, gradient_W2, gradient_b1, gradient_b2, d_output, d_hidden);
            cudaDeviceSynchronize();

            int threads_per_block = 1024;
            int blocks = (HIDDEN_SIZE * INPUT_SIZE + threads_per_block - 1) / threads_per_block;

            update_grads_1<<<blocks, threads_per_block>>>(gradient_W1, gradient_W2, gradient_b1, gradient_b2, d_hidden, d_output, images_d + (i * INPUT_SIZE), hidden_d);
            cudaDeviceSynchronize();

            update_grads_2<<<blocks, threads_per_block>>>(gradient_W1, gradient_W2, gradient_b1, gradient_b2);
            cudaDeviceSynchronize();

            update_net<<<blocks, threads_per_block>>>(nni->device_interface, gradient_W1, gradient_W2, gradient_b1, gradient_b2);
            cudaDeviceSynchronize();
            
            cudaFree(gradient_W1);
            cudaFree(gradient_W2);
            cudaFree(gradient_b1);
            cudaFree(gradient_b2);

            float output[BATCH_SIZE * OUTPUT_SIZE];
            cudaMemcpy(output, output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
            // Compute loss & accuracy
            for (int b = 0; b < BATCH_SIZE; b++) {
                int index = i + b;
                if (index >= numImages) break;
        
                for (int k = 0; k < OUTPUT_SIZE; k++)
                    loss -= labels[(i + b) * OUTPUT_SIZE + k] * log(output[b * OUTPUT_SIZE + k]);
        
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[b * OUTPUT_SIZE + j] > output[b * OUTPUT_SIZE + pred]) pred = j;
                    if (labels[(i + b) * OUTPUT_SIZE + j] > labels[(i + b) * OUTPUT_SIZE + actual]) actual = j;
                }
                if (pred == actual) correct++;
            }

            cudaFree(hidden_d);
            cudaFree(output_d);
            cudaFree(d_hidden);
            cudaFree(d_output);
            cudaFree(sum);
        }        

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    cudaFree(images_d);
    cudaFree(labels_d);
}

void evaluate(NNInterface* nni, float* images, float* labels, int numImages) {
    int correct = 0;
    int size_1 = BATCH_SIZE * HIDDEN_SIZE;
    int size_2 = HIDDEN_SIZE * INPUT_SIZE;
    int threads__2 = 256;
    int blocks_1 = (size_1 + threads__2 - 1) / threads__2;
    int blocks_2 = (size_2 + threads__2 - 1) / threads__2;
    float* images_d;
    cudaMalloc((void**)&images_d, numImages * INPUT_SIZE * sizeof(float));
    cudaMemcpy(images_d, images, numImages * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        
        if(i + BATCH_SIZE >= numImages) { // Safety again
            i = numImages - BATCH_SIZE;
        }

        float* hidden_d;
        float* output_d;
        cudaMalloc((void**)&hidden_d, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
        __half* hidden_d_half;
        __half* W1_half;
        cudaMalloc((void**)&hidden_d_half, BATCH_SIZE * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc((void**)&W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
        convertToHalf<<<blocks_1, threads__2>>>((float*)hidden_d, hidden_d_half, size_1);
        convertToHalf<<<blocks_2, threads__2>>>((float*)nni->device_interface->W1, W1_half, size_2);

        int max_i_in_forward = 128;
        forward<<<BATCH_SIZE, max_i_in_forward>>>(nni->device_interface, images_d + (i * INPUT_SIZE), hidden_d, hidden_d_half, output_d, W1_half);

        float output[BATCH_SIZE * OUTPUT_SIZE];
        cudaMemcpy(output, output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        for (int b = 0; b < BATCH_SIZE; b++) {
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[b * OUTPUT_SIZE + j] > output[b * OUTPUT_SIZE + pred])
                    pred = j;
                if (labels[(i + b) * OUTPUT_SIZE + j] > labels[(i + b) * OUTPUT_SIZE + actual])
                    actual = j;
            }
            if (pred == actual) correct++;
        }
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
}

// Read MNIST dataset
float* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float* images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

float* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float* labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// Free network memory
void freeNetwork(NNInterface* nni) {
    cudaFree(nni->host_interface->W1);
    cudaFree(nni->host_interface->W2);
    cudaFree(nni->host_interface->b1);
    cudaFree(nni->host_interface->b2);
    cudaFree(nni->device_interface);
    free(nni->host_interface);
    free(nni);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    float* train_images = loadMNISTImages("data/train-images.idx3-ubyte", train_amount);
    float* train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", train_amount);
    float* test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", test_amount);
    float* test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", test_amount);

    NNInterface* nni = createNetwork();
    train(nni, train_images, train_labels, train_amount, train_amount);
    evaluate(nni, test_images, test_labels, test_amount);

    freeNetwork(nni);
    return 0;
}