#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

// Edit BATCH_SIZE to simulate for different batches.
// When set to 1 it will have the exact same results as V1.
// Setting to higher values will simulate how the
// Train Accuracy will change.
// Higher Batch values will need to compensate with
// a higher EPOCHS value for achieving the same level of accuracy,
// or a lower learning rate.

// Careful not to set EPOCHS too high to avoid overfitting, and
// careful not to set learning rate too low to avoid stagnation
// EPOCHS should range between 3 and 12
// learning rate should range between 0.1 and 0.001
// Adjust batch size as you see fit with time to train and accuracy.

#define INPUT_SIZE 784 // 28 x 28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 32
#define NUM_CLASSES 10  // Digits 0-9

#define train_amount 60000
#define test_amount 10000

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double* allocateMatrix(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

// Activation functions
__device__ double relu(double x) {
    if(x > 0)
        return x;
    else
        return 0;
}

// Neural network structure
typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

typedef struct { 
    // Exists so that there isn't an illegal access if the host wants to do something with
    // the device based W1, W2, b1, and b2. Like cudaFree or cudaMemcpy.
    // Both have the same NN in them, it's just for host vs device access.
    // nni would still ideally exist on the host though, device shouldn't get that.
    NeuralNetwork* host_interface;
    NeuralNetwork* device_interface;
} NNInterface;

// Initialize neural network
NNInterface* createNetwork() {
    // Steps happening:
    // 1. Make W1, W2, b1, b2 on host and initialize them because rand() on CUDA is not easily available.
    // 2. Copy to device W1, W2, b1, b2.
    // 3. Give those pointers to the NN struct on host.
    //  . Host now has 4 pointers which all belong on device.
    // 4. Copy contents of that NN struct to the one on the device.
    //  . So now there's a device pointer to 4 other device pointers which will be used for everything.

    // Step 1
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    double* W1_h = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    double* W2_h = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    double* b1_h = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* b2_h = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(1234); // Kept the same for testing
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            W1_h[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W2_h[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    // Step 2
    double* W1_d;
    double* W2_d;
    double* b1_d;
    double* b2_d;

    cudaMalloc((void**)&W1_d, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&W2_d, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&b1_d, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&b2_d, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(W1_d, W1_h, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(W2_d, W2_h, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice); // These aren't really required but
    cudaMemcpy(b2_d, b2_h, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice); // better safe than sorry. calloc is used for a reason.

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

// Forward pass
__global__ void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {

    // Conversion from Linear to Batches:
    // input  dimensions are now 784 * BATCH_SIZE
    // hidden dimensions are now 128 * BATCH_SIZE
    // output dimensions are now 10  * BATCH_SIZE

    // <<<BATCH_SIZE, 128>>>
    int b = blockIdx.x;
    int i = threadIdx.x;
    __shared__ double sum;

    // Compute hidden layer (input × W1 + b1)

    // Loop 1: b = 0 to BATCH_SIZE (Multiple of 32)
    // Loop 2: i = 0 to HIDDEN_SIZE (128)
    if(i < HIDDEN_SIZE) {
        hidden[b * HIDDEN_SIZE + i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[b * HIDDEN_SIZE + i] += net->W1[i * INPUT_SIZE + j] * input[b * INPUT_SIZE + j];
        }
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
    if(i == 0) { // atomicAdd for double's wasn't working. One thread finds the sum.
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
__global__ void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, double* gradient_W1, double* gradient_W2, double* gradient_b1, double* gradient_b2, double* d_output, double* d_hidden) {

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
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            sum += net->W2[j * HIDDEN_SIZE + i] * d_output[b * OUTPUT_SIZE + j];
    
        d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0) ? sum : 0.0;
    }
}

__global__ void update_grads_1(double* gradient_W1, double* gradient_W2, double* gradient_b1, double* gradient_b2, double* d_hidden, double* d_output, double* input, double* hidden) {

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

__global__ void update_grads_2(double* gradient_W1, double* gradient_W2, double* gradient_b1, double* gradient_b2) {
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

__global__ void update_net(NeuralNetwork* net, double* gradient_W1, double* gradient_W2, double* gradient_b1, double* gradient_b2) {
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

// Train network
void train(NNInterface* nni, double* images, double* labels, int numImages, int numLabels) {

    double* images_d;
    cudaMalloc((void**)&images_d, numImages * INPUT_SIZE * sizeof(double));
    cudaMemcpy(images_d, images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    double* labels_d;
    cudaMalloc((void**)&labels_d, numLabels * OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(labels_d, labels, numLabels * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            if(i + BATCH_SIZE >= numImages) { // Safety to adjust for total images and batch sizes if it doesn't equally divide
                i = numImages - BATCH_SIZE;
            }
            
            double* hidden_d;
            double* output_d;
            cudaMalloc((void**)&hidden_d, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
            cudaMalloc((void**)&output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));

            double* d_hidden;
            double* d_output;
            cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
            cudaMalloc((void**)&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));

            double* sum;
            cudaMalloc((void**)&sum, sizeof(double));
    
            int max_i_in_forward = 128;
            forward<<<BATCH_SIZE, max_i_in_forward>>>(nni->device_interface, images_d + (i * INPUT_SIZE), hidden_d, output_d);
            cudaDeviceSynchronize();

            double* gradient_W1;
            double* gradient_W2;
            double* gradient_b1;
            double* gradient_b2;

            cudaMalloc((void**)&gradient_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
            cudaMalloc((void**)&gradient_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
            cudaMalloc((void**)&gradient_b1, HIDDEN_SIZE * sizeof(double));
            cudaMalloc((void**)&gradient_b2, OUTPUT_SIZE * sizeof(double));

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

            double output[BATCH_SIZE * OUTPUT_SIZE];
            cudaMemcpy(output, output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        
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
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    cudaFree(images_d);
    cudaFree(labels_d);
}

void evaluate(NNInterface* nni, double* images, double* labels, int numImages) {
    int correct = 0;

    double* images_d;
    cudaMalloc((void**)&images_d, numImages * INPUT_SIZE * sizeof(double));
    cudaMemcpy(images_d, images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        
        if(i + BATCH_SIZE >= numImages) { // Safety again
            i = numImages - BATCH_SIZE;
        }

        double* hidden_d;
        double* output_d;
        cudaMalloc((void**)&hidden_d, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
        cudaMalloc((void**)&output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));

        int max_i_in_forward = 128;
        forward<<<BATCH_SIZE, max_i_in_forward>>>(nni->device_interface, images_d + (i * INPUT_SIZE), hidden_d, output_d);

        double output[BATCH_SIZE * OUTPUT_SIZE];
        cudaMemcpy(output, output_d, BATCH_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

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
    double* images = allocateMatrix(numImages, INPUT_SIZE);
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
    double* labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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

    double* train_images = loadMNISTImages("data/train-images.idx3-ubyte", train_amount);
    double* train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", train_amount);
    double* test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", test_amount);
    double* test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", test_amount);

    NNInterface* nni = createNetwork();
    train(nni, train_images, train_labels, train_amount, train_amount);
    evaluate(nni, test_images, test_labels, test_amount);

    freeNetwork(nni);
    return 0;
}

