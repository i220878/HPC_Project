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
#define BATCH_SIZE 1
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
double relu(double x) {
    if(x > 0)
        return x;
    else
        return 0;
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

// Neural network structure
typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(1234); // Kept the same for testing
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {

    // Conversion from Linear to Batches:
    // input  dimensions are now 784 * BATCH_SIZE
    // hidden dimensions are now 128 * BATCH_SIZE
    // output dimensions are now 10  * BATCH_SIZE

    for (int b = 0; b < BATCH_SIZE; b++) {
        // Compute hidden layer (input × W1 + b1)
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden[b * HIDDEN_SIZE + i] = net->b1[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden[b * HIDDEN_SIZE + i] += net->W1[i * INPUT_SIZE + j] * input[b * INPUT_SIZE + j];
            }
        }

        // Apply ReLU activation
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            if (hidden[b * HIDDEN_SIZE + i] < 0)
                hidden[b * HIDDEN_SIZE + i] = 0.0;
        }

        // Compute output layer (hidden × W2 + b2)
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[b * OUTPUT_SIZE + i] = net->b2[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output[b * OUTPUT_SIZE + i] += net->W2[i * HIDDEN_SIZE + j] * hidden[b * HIDDEN_SIZE + j];
            }
        }
        softmax(&output[b * OUTPUT_SIZE], OUTPUT_SIZE);
    }
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[BATCH_SIZE * OUTPUT_SIZE];
    double d_hidden[BATCH_SIZE * HIDDEN_SIZE];

    double gradient_W1[INPUT_SIZE * HIDDEN_SIZE] = {};
    double gradient_W2[HIDDEN_SIZE * OUTPUT_SIZE] = {};
    double gradient_b1[HIDDEN_SIZE] = {};
    double gradient_b2[OUTPUT_SIZE] = {};

    // Compute d_output = output - target
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            d_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] - target[b * OUTPUT_SIZE + i];
        }
    }

    // Compute d_hidden
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++)
                sum += net->W2[j * HIDDEN_SIZE + i] * d_output[b * OUTPUT_SIZE + j];

            d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0) ? sum : 0.0;
        }
    }

    // Accumulate gradients for W2 and b2
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                gradient_W2[i * HIDDEN_SIZE + j] += d_output[b * OUTPUT_SIZE + i] * hidden[b * HIDDEN_SIZE + j];
            }
            gradient_b2[i] += d_output[b * OUTPUT_SIZE + i];
        }
    }

    // Accumulate gradients for W1 and b1
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                gradient_W1[i * INPUT_SIZE + j] += d_hidden[b * HIDDEN_SIZE + i] * input[b * INPUT_SIZE + j];
            }
            gradient_b1[i] += d_hidden[b * HIDDEN_SIZE + i];
        }
    }

    // Average gradients
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        gradient_W1[i] /= BATCH_SIZE;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        gradient_W2[i] /= BATCH_SIZE;

    for (int i = 0; i < HIDDEN_SIZE; i++)
        gradient_b1[i] /= BATCH_SIZE;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        gradient_b2[i] /= BATCH_SIZE;

    // Update weights and biases
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->W2[i] -= LEARNING_RATE * gradient_W2[i];

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->W1[i] -= LEARNING_RATE * gradient_W1[i];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * gradient_b2[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * gradient_b1[i];
}

// Train network
void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {

            if(i + BATCH_SIZE >= numImages) { // Safety to adjust for total images and batch sizes if it doesn't equally divide
                i = numImages - BATCH_SIZE;
            }
            
            double hidden[BATCH_SIZE * HIDDEN_SIZE];
            double output[BATCH_SIZE * OUTPUT_SIZE];
    
            forward(net, &images[i * INPUT_SIZE], hidden, output);
            backward(net, &images[i * INPUT_SIZE], hidden, output, &labels[i * OUTPUT_SIZE]);
        
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
        }        

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;

    double hidden[BATCH_SIZE * HIDDEN_SIZE];
    double output[BATCH_SIZE * OUTPUT_SIZE];

    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        
        if(i + BATCH_SIZE >= numImages) { // Safety again
            i = numImages - BATCH_SIZE;
        }
        forward(net, &images[i * INPUT_SIZE], hidden, output);
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

    double* train_images = loadMNISTImages("data/train-images.idx3-ubyte", train_amount);
    double* train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", train_amount);
    double* test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", test_amount);
    double* test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", test_amount);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, train_amount);
    evaluate(net, test_images, test_labels, test_amount);

    freeNetwork(net);
    return 0;
}

