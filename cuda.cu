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

__device__ float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;
    
    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }
    
    return sum;
}

__global__ void KNN(ArffData* train, ArffData* test, int k) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int* predictions = (int*)malloc(test->num_instances() * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();

    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)calloc(num_classes, sizeof(int));

    for(int queryIndex = 0; queryIndex < test->num_instances(); queryIndex++) {
        for(int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {
            
            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));

            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float(); // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k;i++){
            classCounts[(int)candidates[2*i+1]] += 1;
        }
        
        int max = -1;
        int max_index = 0;
        for(int i = 0; i < num_classes;i++){
            if(classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;

        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));
    }
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){

    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    int numAttr = train->num_attributes();
    int numClasses = train->num_classes();
    int instanceSizes[2] = {train->num_instances(), test->num_instances()};

    // Allocate host memory
    float (*h_train_instances)[numAttr] = malloc(sizeof(float[instanceSizes[0]][numAttr]));
    float (*h_test_instances)[numAttr] = malloc(sizeof(float[instanceSizes[1]][numAttr]));
    int* h_predictions = (int*) malloc(instanceSizes[1] * sizeof(int));

    for(int i = 0; i < instanceSizes[0]; i++) {
        for(int j = 0; j < numAttr; j++) {
            h_train_instances[i][j] = train->get_instance(i)->get(j)->operator float();

            if(i < instanceSizes[1])
                h_test_instances[i][j] = test->get_instance(i)->get(j)->operator float();
        }
    }

    // Allocate device memory
    float (*d_train_instances)[numAttr];
    float (*d_test_instances)[numAttr];
    int *d_predictions;

    cudaMalloc(&d_train_instances, sizeof(float[instanceSizes[0]][numAttr]));
    cudaMalloc(&d_test_instances, sizeof(float[instanceSizes[1]][numAttr]));
    cudaMalloc(&d_predictions, instanceSizes[1] * sizeof(int));

    // Copy host memory to device memory
    cudaMemcpy(d_train_instances, h_train_instances, sizeof(float[instanceSizes[0]][numAttr]), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_instances, h_test_instances, sizeof(float[instanceSizes[1]][numAttr]), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, instanceSizes[1] * sizeof(int), cudaMemcpyHostToDevice);

    // Configure the blocks and grid sizes
    int threadsPerBlock = 32;
	int gridSize = (instanceSizes[1] + threadsPerBlock - 1) / threadsPerBlock;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Launch the kernel function
    // KNN<<<gridSize, threadsPerBlock>>>(train, test, k);
    
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

    // Transfer device results to host memory

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    printf("The %i-NN classifier for %lu test instances on %lu train instances required %f ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), milliseconds, accuracy);

    // Free memory
    cudaFree(d_train_instances);
	cudaFree(d_test_instances);
	cudaFree(d_predictions);
	free(h_train_instances);
	free(h_test_instances);
	free(h_predictions);

    return 0;
}
