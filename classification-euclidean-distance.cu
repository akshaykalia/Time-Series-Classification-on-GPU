#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <cuda_profiler_api.h>
#include "helper.h"
#include <float.h>


/////////////////////////////////////////////////////////////////////////
// Init
/////////////////////////////////////////////////////////////////////////

uint64_t noOfTimeSeries = 20;
uint64_t lenOfTimeSeries = 70;
uint64_t noOfTestTimeSeries = 10;
/////////////////////////////////////////////////////////////////////////
  
void usage(){
    printf("********************************\n");
    printf("************* USAGE ************\n");
    printf("********************************\n");
    printf("./classification-ed [training-file] [number-of-time-series] [length-of-time-series] [testing-file] [number-of-times-series-in-test]\n");
    printf("eg. ./classification-ed SonyAIBORobotSurface_TRAIN 20 70 \n");
    printf("********************************\n");
}

void readfile(char* inputFileName,float* _data,int* _class,uint64_t len)
{
    
    std::ifstream in_file;
    in_file.open(inputFileName);
    if(!in_file) {
        printf("\nFile Not Found !");
        exit(1);
    }

    float class_in;
    float data_in;
    
    long int i, j;
    for(i=0; i<len; i++)
    {
        in_file >> class_in;
        _class[i] = (int)class_in;
        //printf("class : %d\n",_class[i]);
        
        for (j=0; j<lenOfTimeSeries; j++)
        {
            in_file >> data_in;
            _data[i*lenOfTimeSeries+j] = data_in;
            //printf("%f, ",_data[i*lenOfTimeSeries+j]);
        }
        //printf("\n");
    }
    in_file.close();
    
}

////////////////////////////////////////////////////////////////

__device__ void normalize(float* d_data, float mean, float stdev, uint64_t t, float* norm_data, const int L)
{
    int i = 0;
    for(i=0; i<L; i++)
    {
        norm_data[i] = (d_data[t+i]-mean)/stdev;
    }
}

////////////////////////////////////////////////////////////////

__global__ void Euclidean_Distance(float* trainingData, float* testData, float* output, int length)
{    
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp[1024];

    float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
    
    int t = length*(idx/length) + idx%length;
    float tempLoc;
    for(i=t; i<t+length; i++)
    {
        tempLoc = trainingData[i];
        sum += tempLoc;
        sum_sqr += tempLoc * tempLoc;
    }

    mean = sum / length;
    mean_sqr = mean*mean;
            
    variance = (sum_sqr/length) - mean_sqr;
    std_dev = sqrt(variance);

    i = 0;
    for(i=0; i<length; i++){
        temp[i] = (trainingData[t+i]-mean) / std_dev; 
    }

    
    float errorSummation  = 0;
    for(i=0; i < length; i++)
    {
        errorSummation += (temp[i] - testData[i])*(temp[i] - testData[i]); 
    }
    errorSummation = sqrt(errorSummation);
    output[idx] = errorSummation;
}

int main(int argc, char * argv[])
{
    clock_t start, end;
    fprintf(stderr, "Initializing ... \n"); 
    char* inputFileName = argv[1];
    int isDefault = 0;
    if(!inputFileName){
        printf("No test file provided. Using default file : SonyAIBORobotSurface_TRAIN\n");
        inputFileName = "SonyAIBORobotSurface_TRAIN";
        isDefault = 1;
    }
    if(argc > 1){
        noOfTimeSeries = atoi(argv[2]);
    }else{
        if(isDefault == 0){
            printf("Number of time series not provided. Exiting\n");
            exit(0);
        }
    }
    if(argc > 2){
        lenOfTimeSeries = atoi(argv[3]);
    }
    else{
        if(isDefault == 0){
            printf("Length of time series not provided. Exiting\n");
            exit(0);
        }
    }

    uint64_t train_size = noOfTimeSeries * lenOfTimeSeries * sizeof(float);
    uint64_t test_size;// = noOfTestTimeSeries * lenOfTimeSeries * sizeof(float);
    
    //storage allocation for train data and train class labels
    float* train_data = (float*) malloc(train_size);
    int* train_class = (int *) malloc(noOfTimeSeries*sizeof(int));
    //storage allocation for test data and test class labels
    float* test_data;// = (float*) malloc (test_size);
    int* test_class;// = (int *) malloc(noOfTestTimeSeries * sizeof(int));
    
    //get training file
    printf("Reading train file\n");
    //read training file
    readfile(inputFileName, train_data, train_class, noOfTimeSeries);
    printf("===================================================\n");
    printf("Training File : %s\n",inputFileName);
    printf("Number of Time Series : %d\n",noOfTimeSeries);
    printf("Length of Time Series : %d\n",lenOfTimeSeries);
    
    // If Testing File is provided
    if(argc == 6 || isDefault == 1){
        char* testFileName;
        if(isDefault == 0){
            testFileName = argv[4];
            noOfTestTimeSeries = atoi(argv[5]);
        }else{
            testFileName = "SonyAIBORobotSurface_TEST";
            noOfTestTimeSeries = 601;
        }
        printf("----------------------------------------------------\n");
        //get testing file
        printf("Reading test file\n");
        test_size = noOfTestTimeSeries * lenOfTimeSeries * sizeof(float);
        test_data = (float*) malloc (test_size);
        test_class = (int *) malloc(noOfTestTimeSeries * sizeof(int));
        //read test file
        readfile(testFileName, test_data, test_class, noOfTestTimeSeries);
        
        printf("Testing File : %s\n",testFileName);
        printf("Number of Time Series to validate: %d\n",noOfTestTimeSeries);
    }
    
    int minNumberOfThreads = 1024;
    //if(argc > 6){
    //    minNumberOfThreads = atoi(argv[6]);
    //}
    printf("===================================================\n");
    checkCudaErrors(cudaDeviceReset());
    cudaProfilerStart();
    //GPU number present in the system
    //int noOfGPUs;
    //checkCudaErrors(cudaGetDeviceCount(&noOfGPUs));
    //printf("Total GPUs on System : %d\n", noOfGPUs);
    int i = 0;
    int threadsPerBlock = min((int)ceil(lenOfTimeSeries/(float)32)*32,minNumberOfThreads);
    int noOfBlocks = ceil((noOfTimeSeries*lenOfTimeSeries)/(float)threadsPerBlock);
    printf("noOfBlocks %d threadsPerBlock %d\n",noOfBlocks ,threadsPerBlock);

    float* d_subseq = 0;
    checkCudaErrors(cudaMalloc((void**)&d_subseq, train_size));
    float* d_test_series = 0;
    checkCudaErrors(cudaMalloc((void**)&d_test_series,lenOfTimeSeries*sizeof(float)));
    float* d_train_data = 0;
    checkCudaErrors(cudaMalloc((void**)&d_train_data, train_size));
    checkCudaErrors(cudaMemcpy(d_train_data, train_data, train_size, cudaMemcpyHostToDevice));
    
    
    start = clock();
    int errorCount = 0 , minIndex  = -1;
    cudaStream_t streams[noOfTestTimeSeries];

    for (i=0;i < noOfTestTimeSeries;i++)
    {
        float* subseq = (float*)malloc(train_size);
        checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(cudaMemcpyAsync(d_test_series, test_data+(lenOfTimeSeries*i), lenOfTimeSeries*sizeof(float), cudaMemcpyHostToDevice,streams[i])); 
        Euclidean_Distance<<<noOfBlocks, threadsPerBlock, 0, streams[i]>>>(d_train_data, d_test_series, d_subseq, lenOfTimeSeries);
        checkCudaErrors(cudaMemcpyAsync(subseq, d_subseq, train_size, cudaMemcpyDeviceToHost, streams[i]));
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        float minDistance = FLT_MAX;
        minIndex = -1; 
        int j = 0;
        for(; j < noOfTimeSeries ; j++ )
        {
              if ( minDistance > subseq[j*lenOfTimeSeries] )
              {
                 minDistance = subseq[j*lenOfTimeSeries];
                 minIndex = j;
              }
             
        }
      
        if( train_class[minIndex] != test_class[i] )
            errorCount++;
        free(subseq);
        printf("%d\t%d\t %d\t%d\t%3.6f\n",i , test_class[i] ,train_class[minIndex], minIndex , minDistance );
    }

    checkCudaErrors(cudaFree(d_train_data));
    checkCudaErrors(cudaFree(d_subseq));
    checkCudaErrors(cudaFree(d_test_series));
    
    free(train_class);
    free(test_class);
    
    end = clock() - start;
    double endtime = (double)end / ((double)CLOCKS_PER_SEC);
    printf("Total Time GPU : %f\n", endtime);
    printf("Accuracy is %f\n",(float)(noOfTestTimeSeries-errorCount)*(100.0/noOfTestTimeSeries));    
    cudaProfilerStop();
    checkCudaErrors(cudaDeviceReset());
    return 0;
}
