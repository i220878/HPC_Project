#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784 // 28 x 28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9
#define train_amount 60000
#define test_amount 10000

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}
// Allocate memory for a matrix
float* allocateMatrix_1D(int rows, int cols) {
    return (float*)malloc(rows * cols * sizeof(float));
}
// Allocate memory for a matrix
float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)malloc(cols * sizeof(float));
    }
    return mat;
}

// Activation functions
void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(float* x, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
//////////////////////////////////////////////////////////////
//GPU Activation functions
__global__ void relu_GPU (float* x)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < HIDDEN_SIZE)
        x[tid] = (x[tid] > 0) ? x[tid] : 0;
}

__global__ void softmax_GPU(float* x, int size)    //just doing serial execution for this....
{
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

/////////////////////////////////////////////////////////////
// Neural network structure
typedef struct {
    float* W1;
    float* W2;
    float* b1;
    float* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix_1D(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix_1D(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] = ((float)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] = ((float)rand() / RAND_MAX) * 0.01;

    return net;
}

// Forward pass
void forward(NeuralNetwork* net, float* input, float* hidden, float* output) {
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

// Backpropagation
void backward(NeuralNetwork* net, float* input, float* hidden, float* output, float* target) {
    float d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}
//***************************************************************************************************
/////////////////////////////////////////////
//foward pass GPU version
//dividing the oringinal forward function into two parts , hidden and output (division accroding to the layers)...
__global__ void forward_hidden_GPU(float* b1, float* w1, float* input, float* hidden)
{
    int t_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (t_id < HIDDEN_SIZE)
    {
        hidden[t_id] = b1[t_id];
        for (int k = 0; k < INPUT_SIZE; k++)
            hidden[t_id] += w1[t_id*INPUT_SIZE + k] * input[k];
    }
}
__global__ void forward_output_GPU(float* b2, float* w2, float* hidden, float* output)
{
    int t_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (t_id < OUTPUT_SIZE)
    {
        output[t_id] = b2[t_id];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[t_id] += w2[t_id*HIDDEN_SIZE + j] * hidden[j];
    }
}
//The MAIN FUNCTION for the forward functionality....
//Declarations of the required helper functions of the main FORWARD FUNCTION....

// Compute output layer gradient
__global__ void compute_output_gradient_GPU(float* output, float* target, float* d_output, int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < size) {
        d_output[tid] = output[tid] - target[tid];
    }
}

// Compute hidden layer gradient
__global__ void compute_hidden_gradient_GPU(float* W2, float* d_output, float* hidden, float* d_hidden, int hidden_size, int output_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < hidden_size) {
        d_hidden[tid] = 0;
        for (int j = 0; j < output_size; j++) {
            d_hidden[tid] += W2[j*hidden_size + tid] * d_output[j];
        }
        d_hidden[tid] *= (hidden[tid] > 0); // ReLU derivative
    }
}

// Update output weights
__global__ void update_output_weights_GPU(float* W2, float* b2, float* d_output, float* hidden, int output_size, int hidden_size, float learning_rate) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i < output_size && j < hidden_size) {
        W2[i*hidden_size + j] -= learning_rate * d_output[i] * hidden[j];
    }
    if (j == 0 && i < output_size) {
        b2[i] -= learning_rate * d_output[i];
    }
}

// Update hidden weights
__global__ void update_hidden_weights_GPU(float* W1, float* b1, float* d_hidden, float* input, int hidden_size, int input_size, float learning_rate) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if (i < hidden_size && j < input_size) {
        W1[i*input_size + j] -= learning_rate * d_hidden[i] * input[j];
    }
    
    if (j == 0 && i < hidden_size) {
        b1[i] -= learning_rate * d_hidden[i];
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void ptr_to_array_1D(float* ptrr, int size, float array[])
{
    for (int i=0; i<size; i++)
        array[i] = ptrr[i];
}
void array_to_ptr_1D(float* ptr, int size, float array[])
{
    for (int i=0; i<size; i++)
        ptr[i] = array[i];
}
void ptr_to_ptr_1D(float* source, int size, float* dest)
{
    for (int i=0; i<size; i++)
        dest[i] = source[i];
}



//***************************************************************************************************

// Train network
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    //////////////////////////////////////////////////////////////////////////////////////
    //Allocating all the device memory in this function
    //NeuralNetwork* d_net;                      //neural net for the device...
    //float w1[HIDDEN_SIZE*OUTPUT_SIZE], w2[OUTPUT_SIZE*HIDDEN_SIZE], b1[HIDDEN_SIZE], b2[OUTPUT_SIZE];
    float* w1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* w2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *d_w1, *d_w2, *d_b1, *d_b2;   //device side of the neural networks variables
    ptr_to_ptr_1D(net->W1, HIDDEN_SIZE * INPUT_SIZE, w1);
    ptr_to_ptr_1D(net->W2, OUTPUT_SIZE * HIDDEN_SIZE, w2);
    ptr_to_ptr_1D(net->b1, HIDDEN_SIZE, b1);
    ptr_to_ptr_1D(net->b2, OUTPUT_SIZE, b2);
    float *d_input, *d_hidden, *d_output;    //remaining variables for device...
    //Allocating device memory...
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));          //the image is of size (num_images*INPUT_SIZE), and for each image this function is called (in a for loop), so the dimension of input = INPUT_SIZE. SEE the TRAIN function for more detailed understanding....
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));        //this is the size of hidden array, see the "train" function to confirm this...
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));        //this is the size of output array, see the "train" function to confirm this...
    cudaMalloc(&d_w1, HIDDEN_SIZE * INPUT_SIZE*sizeof(float));
    cudaMalloc(&d_w2, OUTPUT_SIZE * HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE*sizeof(float));
    cudaMemcpy(d_w1, w1, HIDDEN_SIZE*INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    //////for backwarfd function
    float* d_label;
    cudaMalloc(&d_label, OUTPUT_SIZE * sizeof(float));
    float backwards_h_output[OUTPUT_SIZE], backwards_h_hidden[HIDDEN_SIZE];
    float *backwards_d_output, *backwards_d_hidden;
    cudaMalloc(&backwards_d_output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&backwards_d_hidden, HIDDEN_SIZE * sizeof(float));
    int threads = 128;
    int blocks_hidden = (HIDDEN_SIZE + threads - 1) / threads;
    int blocks_output = (OUTPUT_SIZE + threads - 1) / threads;
    ///////////////////////////////////////////////////////////////////////////////////////
    int blocks_hiddenSize = (HIDDEN_SIZE + threads - 1)/threads;       //for ceiling purpose...
    int blocks_outputSize = (OUTPUT_SIZE + threads - 1)/threads; 

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            //******************************************************************************************************** */
            float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            //copying host data to device//
            cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_hidden, hidden, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_output, output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            //making function calls of forward function
            //now making all required kernel calls....
            forward_hidden_GPU<<<blocks_hiddenSize, threads>>>(d_b1, d_w1, d_input, d_hidden);
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            cudaDeviceSynchronize();
            relu_GPU<<<blocks_hiddenSize, threads>>>(d_hidden);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            cudaDeviceSynchronize();
            //threads will remian same, blocks will be different
            forward_output_GPU<<<blocks_outputSize, threads>>>(d_b2, d_w2, d_hidden, d_output);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            cudaDeviceSynchronize();
            //perform the remianing function in serial on GPU...
            softmax_GPU<<<1, 1>>>(d_output, OUTPUT_SIZE);
            cudaDeviceSynchronize();
            ////////////////////////////////////////////////////////////////////////////
            cudaMemcpy(d_label, labels[i], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(backwards_d_output, backwards_h_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(backwards_d_hidden, backwards_h_hidden, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            //printf("Came heere!!!!!!!!!!!!!!!!!!!!----%i-----\n", i);
            //fflush(stdout);
            // Backward pass
            compute_output_gradient_GPU<<<blocks_output, threads>>>(d_output, d_label, backwards_d_output, OUTPUT_SIZE);
            cudaDeviceSynchronize();
            //printf("Came heere!!!!!!!!!!!!!!!!!!!!----%i-----\n", i);
            //fflush(stdout);
            // err = cudaGetLastError();
            //  if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //      exit(1);
            // }
            compute_hidden_gradient_GPU<<<blocks_hidden, threads>>>(d_w2, backwards_d_output, d_hidden, backwards_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
            cudaDeviceSynchronize();
            //printf("Came heere!!!!!!!!!!!!!!!!!!!!----%i-----\n", i);
            //fflush(stdout);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            update_output_weights_GPU<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(d_w2, d_b2, backwards_d_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
            cudaDeviceSynchronize();
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            update_hidden_weights_GPU<<<HIDDEN_SIZE, INPUT_SIZE>>>(d_w1, d_b1, backwards_d_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
            cudaDeviceSynchronize();
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //printf("Came heere!!!!!!!!!!!!!!!!!!!!----%i-----\n", i);
            //fflush(stdout);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("CUDA error: %s\n", cudaGetErrorString(err));
            //     exit(1);
            // }
            //cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            //******************************************************************************************************** */
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, float** images, float** labels, int numImages) {
    ///////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //Allocating all the device memory in this function
    //NeuralNetwork* d_net;                      //neural net for the device...
    //float w1[HIDDEN_SIZE*OUTPUT_SIZE], w2[OUTPUT_SIZE*HIDDEN_SIZE], b1[HIDDEN_SIZE], b2[OUTPUT_SIZE];
    float* w1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* w2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *d_w1, *d_w2, *d_b1, *d_b2;   //device side of the neural networks variables
    ptr_to_ptr_1D(net->W1, HIDDEN_SIZE * INPUT_SIZE, w1);
    ptr_to_ptr_1D(net->W2, OUTPUT_SIZE * HIDDEN_SIZE, w2);
    ptr_to_ptr_1D(net->b1, HIDDEN_SIZE, b1);
    ptr_to_ptr_1D(net->b2, OUTPUT_SIZE, b2);
    float *d_input, *d_hidden, *d_output;    //remaining variables for device...
    //Allocating device memory...
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));          //the image is of size (num_images*INPUT_SIZE), and for each image this function is called (in a for loop), so the dimension of input = INPUT_SIZE. SEE the TRAIN function for more detailed understanding....
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));        //this is the size of hidden array, see the "train" function to confirm this...
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));        //this is the size of output array, see the "train" function to confirm this...
    cudaMalloc(&d_w1, HIDDEN_SIZE * INPUT_SIZE*sizeof(float));
    cudaMalloc(&d_w2, OUTPUT_SIZE * HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE*sizeof(float));
    cudaMemcpy(d_w1, w1, HIDDEN_SIZE*INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 128;
    //////////////////////////////////////////////////////////////////////////////
    int blocks_outputSize = (OUTPUT_SIZE + threads - 1)/threads; 
    int blocks_hiddenSize = (HIDDEN_SIZE + threads - 1)/threads;       //for ceiling purpose...
    int correct = 0;
    cudaError_t err = cudaGetLastError();
    for (int i = 0; i < numImages; i++) {
        //////////////////////////////////////////////////////////////////
        float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        //copying host data to device//
        cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden, hidden, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        forward_hidden_GPU<<<blocks_hiddenSize, threads>>>(d_b1, d_w1, d_input, d_hidden);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     exit(1);
        // }
        cudaDeviceSynchronize();
        relu_GPU<<<blocks_hiddenSize, threads>>>(d_hidden);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     exit(1);
        // }
        cudaDeviceSynchronize();
        forward_output_GPU<<<blocks_outputSize, threads>>>(d_b2, d_w2, d_hidden, d_output);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     exit(1);
        // }
        cudaDeviceSynchronize();
        softmax_GPU<<<1, 1>>>(d_output, OUTPUT_SIZE);
        cudaDeviceSynchronize();
        cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        ///////////////////////////////////////////////////////////////////
        //forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
}

// Read MNIST dataset
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}


float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// Free network memory
void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    float** train_images = loadMNISTImages("data/train-images.idx3-ubyte", train_amount);
    float** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", train_amount);
    float** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", test_amount);
    float** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", test_amount);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, train_amount);
    evaluate(net, test_images, test_labels, test_amount);

    freeNetwork(net);
    return 0;
}
