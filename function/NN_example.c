#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "NN_example.h"

// initialize the neural network
void initialize() {
    for (int i = 0; i < INPUT_SIZE; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // initialize the weight randomly
    }
    bias = ((double)rand() / RAND_MAX) * 2 - 1; // initialize the bias randomly
}

// feedforward propagation
double predict(double inputs[]) {
    double output = 0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        output += weights[i] * inputs[i];
    }
    output += bias;
    return activation(output);
}

// training
void train(double inputs[], double target) {
    double prediction = predict(inputs);
    double error = target - prediction;
    for (int i = 0; i < INPUT_SIZE; i++) {
        weights[i] += LEARNING_RATE * error * inputs[i];
    }
    bias += LEARNING_RATE * error;
}

int main() {
    // initialize the neural network
    initialize();

    // training data
    double training_data[][INPUT_SIZE] = {{0, 0.01}, {0.99, 1}, {1.98, 2.01}, {3.01, 3}};
    double targets[] = {0, 1, 2, 3};

    // test data
    double test_data[][INPUT_SIZE] = {{2.1, 2}, {3, 3.1}, {4.1, 4.0}, {5, 4.99}};
    double test_targets[] = {2, 3, 4, 5};
    // training the Neural network
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < sizeof(training_data) / sizeof(training_data[0]); i++) {
            train(training_data[i], targets[i]);
        }
    }

    // test the NN
    for (int i = 0; i < sizeof(training_data) / sizeof(training_data[0]); i++) {
        double prediction = predict(training_data[i]);
        printf("Input: [%lf, %lf], Target: %lf, Prediction: %lf\n", 
               test_data[i][0], test_data[i][1], test_targets[i], prediction);
    }

    return 0;
}
