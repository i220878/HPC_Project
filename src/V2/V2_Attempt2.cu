#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

#define INPUT_SIZE 784 // 28 x 28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

#define MAX_THREADS 128

#define train_amount 60000
#define test_amount 10000

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Neural network structure
typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

__global__ void initializeNetwork(NeuralNetwork* net, double* W1_d, double* W2_d, double* b1_d, double* b2_d) {
    int id = threadIdx.x;

    if(id == 0) {
        net->W1 = W1_d;
        net->W2 = W2_d;
        net->b1 = b1_d;
        net->b2 = b2_d;
    }

    curandState state;
    curand_init(1234, id, 0, &state);

    __syncthreads();
    
    for (int j = 0; j < INPUT_SIZE; j++)
        net->W1[id * INPUT_SIZE + j] = curand_uniform(&state) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->W2[i * HIDDEN_SIZE + id] = curand_uniform(&state) * 0.01;
}

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net;
    cudaMalloc((void**)&net, sizeof(NeuralNetwork));

    double *W1_d, *W2_d, *b1_d, *b2_d;
    cudaMalloc((void**)&W1_d, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&W2_d, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&b1_d, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&b2_d, OUTPUT_SIZE * sizeof(double));

    initializeNetwork<<<1, MAX_THREADS>>>(net, W1_d, W2_d, b1_d, b2_d);
    cudaDeviceSynchronize();

    return net;
}

// Forward pass
__global__ void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    __shared__ double sum;
    int i = threadIdx.x;

    hidden[i] = net->b1[i];
    for (int j = 0; j < INPUT_SIZE; j++)
        hidden[i] += net->W1[i * INPUT_SIZE + j] * input[j];
        
    hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;

    if(i >= OUTPUT_SIZE)
        return;

    output[i] = net->b2[i];
    // if(i == 0) {
    //     printf("output[0]: %f\n", output[i]);
    // }
    for (int j = 0; j < HIDDEN_SIZE; j++)
        output[i] += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
    output[i] = exp(output[i]);
    // if(i == 0) {
    //     printf("output[0]: %f\n", output[i]);
    // }
    
    // atomicAdd was not working for double.
    __syncthreads();
    if(i == 0) {
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            sum += output[j];
        }
        // printf("sum: %f\n", sum);
    }
    __syncthreads();
    output[i] /= sum;
}

// Backpropagation
__global__ void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    int i = threadIdx.x;

    __shared__ double d_output[OUTPUT_SIZE];
    __shared__ double d_hidden[HIDDEN_SIZE];

    if(i < OUTPUT_SIZE)
        d_output[i] = output[i] - target[i];

    __syncthreads();

    d_hidden[i] = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++)
        d_hidden[i] += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
    d_hidden[i] *= (hidden[i] > 0);

    __syncthreads();
    
    for (int i2 = 0; i2 < OUTPUT_SIZE; i2++)
        net->W2[i2 * HIDDEN_SIZE + i] -= LEARNING_RATE * d_output[i2] * hidden[i];

    for (int j = 0; j < INPUT_SIZE; j++)
        net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];

    if(i < OUTPUT_SIZE)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

__global__ void remaining(double* loss, int* correct, double* labels_d, double* output, int i) {
    // Compute loss & accuracy
    for (int k = 0; k < OUTPUT_SIZE; k++) 
        (*loss) -= labels_d[i * OUTPUT_SIZE + k] * log(output[k]);

    int pred = 0, actual = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        if (output[j] > output[pred]) 
            pred = j;
        if (labels_d[i * OUTPUT_SIZE + j] > labels_d[i * OUTPUT_SIZE + actual]) 
            actual = j;
    }
    if (pred == actual) 
        (*correct)++;
}

// Train network
void train(NeuralNetwork* net, double* images, double* labels, int numImages, int numLabels) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        double* loss;
        int* correct;
        cudaMalloc((void**)&loss, sizeof(double));
        cudaMalloc((void**)&correct, sizeof(int));
        double loss_h = 0.0;
        int correct_h = 0;
        cudaMemcpy(loss, &loss_h, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(correct, &correct_h, sizeof(int), cudaMemcpyHostToDevice);

        int tperb = MAX_THREADS;
        int blocks = 1;


        double* images_d;
        cudaMalloc((void**)&images_d, numImages * INPUT_SIZE * sizeof(double));
        cudaMemcpy(images_d, images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

        double* labels_d;
        cudaMalloc((void**)&labels_d, numLabels * OUTPUT_SIZE * sizeof(double));
        cudaMemcpy(labels_d, labels, numLabels * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

        clock_t epoch_start = clock();
        for (int i = 0; i < numImages; i++) {
            double* hidden;
            double* output;
            cudaMalloc((void**)&hidden, HIDDEN_SIZE * sizeof(double));
            cudaMalloc((void**)&output, OUTPUT_SIZE * sizeof(double));

            forward<<<blocks, tperb>>>(net, &images_d[i * INPUT_SIZE], hidden, output);
            cudaDeviceSynchronize();
            backward<<<blocks, tperb>>>(net, &images_d[i * INPUT_SIZE], hidden, output, &labels_d[i * OUTPUT_SIZE]);
            cudaDeviceSynchronize();
            remaining<<<1, 1>>>(loss, correct, labels_d, output, i);
            cudaDeviceSynchronize();

            cudaFree(hidden);
            cudaFree(output);
        }

        cudaMemcpy(&loss_h, loss, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&correct_h, correct, sizeof(int), cudaMemcpyDeviceToHost);

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss_h / numImages, (correct_h / (double)numImages) * 100, get_time(epoch_start));

        cudaFree(images_d);
        cudaFree(labels_d);
        cudaFree(loss);
        cudaFree(correct);
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}
void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
void forward_cpu(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i * INPUT_SIZE + j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;

    NeuralNetwork* net_cpu = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    cudaMemcpy(net_cpu, net, sizeof(NeuralNetwork), cudaMemcpyDeviceToHost);
    double *W1_CPU, *W2_CPU, *b1_CPU, *b2_CPU;
    W1_CPU = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    W2_CPU = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    b1_CPU = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    b2_CPU = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(W1_CPU, net_cpu->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2_CPU, net_cpu->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1_CPU, net_cpu->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2_CPU, net_cpu->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    net_cpu->W1 = W1_CPU;
    net_cpu->W2 = W2_CPU;
    net_cpu->b1 = b1_CPU;
    net_cpu->b2 = b2_CPU;

    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        
        forward_cpu(net_cpu, &images[i * INPUT_SIZE], hidden, output);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double* images = (double*)malloc(numImages * INPUT_SIZE * sizeof(double));
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


double* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double* labels = (double*)malloc(numLabels * OUTPUT_SIZE * sizeof(double));
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
void freeNetwork(NeuralNetwork* net) {
    // cudaFree(net->W1);
    // cudaFree(net->W2);
    // cudaFree(net->b1);
    // cudaFree(net->b2);
    cudaFree(net);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double* train_images = loadMNISTImages("/home/woha/Desktop/HPC/Project V2/src/data/train-images.idx3-ubyte", train_amount);
    double* train_labels = loadMNISTLabels("/home/woha/Desktop/HPC/Project V2/src/data/train-labels.idx1-ubyte", train_amount);
    double* test_images = loadMNISTImages("/home/woha/Desktop/HPC/Project V2/src/data/t10k-images.idx3-ubyte", test_amount);
    double* test_labels = loadMNISTLabels("/home/woha/Desktop/HPC/Project V2/src/data/t10k-labels.idx1-ubyte", test_amount);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, train_amount, train_amount);
    evaluate(net, test_images, test_labels, test_amount);

    freeNetwork(net);
    return 0;
}

