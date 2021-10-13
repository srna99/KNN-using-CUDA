#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

using namespace std;

__device__ float distance(float *a, float *b, int size) {
    float sum = 0;

    for(int i = 0; i < size - 1; i++) {
        float diff = *(a + i) - *(b + i);
        sum += diff * diff;
    }

    return sum;
}

__global__ void KNN(float *train, float *test, int *predictions, float *candidates, int *classCounts, int *datasetSizes , int k, int numAttr, int numClasses) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < datasetSizes[1]) {
        int candidateForTid = tid * 2 * k;

        for (int keyIndex = 0; keyIndex < datasetSizes[0] * numAttr; keyIndex += numAttr) {
            float dist = distance(test + tid * numAttr, train + keyIndex, numAttr);

            // Add to our candidates
            for (int c = 0; c < k; c++) {
                if (dist < candidates[candidateForTid + 2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[candidateForTid + 2 * x + 2] = candidates[candidateForTid + 2 * x];
                        candidates[candidateForTid + 2 * x + 3] = candidates[candidateForTid + 2 * x + 1];
                    }

                    // Set key vector as potential k NN
                    candidates[candidateForTid + 2 * c] = dist;
                    candidates[candidateForTid + 2 * c + 1] = train[keyIndex + numAttr - 1];

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        int classOfCandidate;

        for (int i = 0; i < k; i++) {
            classOfCandidate = (int) candidates[candidateForTid + 2 * i + 1];
            classCounts[classOfCandidate + tid * numClasses] += 1;
        }

        int max = -1;
        int max_index = 0;
        for (int i = 0; i < numClasses; i++) {
            if (classCounts[i + tid * numClasses] > max) {
                max = classCounts[i + tid * numClasses];
                max_index = i;
            }
        }

        predictions[tid] = max_index;
    }
}

int *computeConfusionMatrix(int *predictions, ArffData *dataset) {
    int *confusionMatrix = (int *) calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for (int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int *confusionMatrix, ArffData *dataset) {
    int successfulPredictions = 0;

    for (int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float)dataset->num_instances();
}

int main(int argc, char *argv[]) {
    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    int numAttr = train->num_attributes();
    int numClasses = train->num_classes();
    int datasetSizes[2] = {(int)train->num_instances(), (int)test->num_instances()};

    // Allocate host memory
    float *h_train_instances = (float *) malloc(datasetSizes[0] * numAttr * sizeof(float));
    float *h_test_instances = (float *) malloc(datasetSizes[1] * numAttr * sizeof(float));
    int *h_predictions = (int *) malloc(datasetSizes[1] * sizeof(int));
    float *h_candidates = (float *) malloc(datasetSizes[1] * k * 2 * sizeof(float));

    for (int i = 0; i < datasetSizes[0]; i++) {
        for (int j = 0; j < numAttr; j++) {
            h_train_instances[i * numAttr + j] = train->get_instance(i)->get(j)->operator float();

            if (i < datasetSizes[1])
                h_test_instances[i * numAttr + j] = test->get_instance(i)->get(j)->operator float();
        }
    }

    for (int i = 0; i < datasetSizes[1] * 2 * k; i++) {
        h_candidates[i] = FLT_MAX;
    }

    // Allocate device memory
    float *d_train_instances, *d_test_instances;
    int *d_predictions;
    float *d_candidates; 
    int *d_class_counts, *d_dataset_sizes; 

    cudaMalloc(&d_train_instances, datasetSizes[0] * numAttr * sizeof(float));
    cudaMalloc(&d_test_instances, datasetSizes[1] * numAttr * sizeof(float));
    cudaMalloc(&d_predictions, datasetSizes[1] * sizeof(int));
    cudaMalloc(&d_candidates, datasetSizes[1] * k * 2 * sizeof(float));
    cudaMalloc(&d_class_counts, datasetSizes[1] * numClasses * sizeof(int));
    cudaMalloc(&d_dataset_sizes, 2 * sizeof(int));

    cudaMemset(d_class_counts, 0, datasetSizes[1] * numClasses * sizeof(int));

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Copy host memory to device memory
    cudaMemcpy(d_train_instances, h_train_instances, datasetSizes[0] * numAttr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_instances, h_test_instances, datasetSizes[1] * numAttr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, datasetSizes[1] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, datasetSizes[1] * k * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset_sizes, datasetSizes, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Configure with different block and grid sizes
    int threadsPerBlock = 1024;
    int gridSize = (datasetSizes[1] + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel function
    KNN<<<gridSize, threadsPerBlock>>>(d_train_instances, d_test_instances, d_predictions, d_candidates, d_class_counts, d_dataset_sizes, k, numAttr, numClasses);

    // Transfer device results to host memory
    cudaMemcpy(h_predictions, d_predictions, datasetSizes[1] * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();
  
    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(h_predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    printf("The %i-NN classifier for %d test instances on %d train instances required %f ms CPU time. Accuracy was %.4f\n", k, datasetSizes[1], datasetSizes[0], milliseconds, accuracy);

    // Free memory
    cudaFree(d_train_instances);
    cudaFree(d_test_instances);
    cudaFree(d_predictions);
    cudaFree(d_dataset_sizes);
    cudaFree(d_candidates);
    cudaFree(d_class_counts);
    free(h_train_instances);
    free(h_test_instances);
    free(h_predictions);
    free(h_candidates);

    return 0;
}
