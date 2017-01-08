#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <limits.h>
#include <float.h>
#include <cuda_profiler_api.h>
#include "helper.h"


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
    printf("./knn [training-file] [number-of-time-series] [length-of-time-series] [testing-file] [number-of-times-series-in-test] [window_size]\n");
    printf("eg. ./knn SonyAIBORobotSurface_TRAIN 20 70 \n");
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


__global__ void NormalizeTimeSeries(float* series,const int totalTimeSeries, const int length, float* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < totalTimeSeries){
        int i=0;
        float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
        int t = idx * length;
        //printf("Normalize Train Series #%d start from index : %d\n",idx,t);
        for(i=t; i<t+length; i++)
        {
            sum += series[i];
            sum_sqr += series[i] * series[i];
        }

        mean = sum / length;
        mean_sqr = mean*mean;
            
        variance = (sum_sqr/length) - mean_sqr;
        std_dev = sqrt(variance);
        i = 0;
        for(i= t; i<t + length; i++){
            series[i] = (series[i]-mean) / std_dev; 
            output[i] = (series[i]-mean) / std_dev;
        }
    }
}

__global__ void DTWDistance(float* test_data, float* train_data, int length, 
    int window, const int numberToCalc, const int current,  
    float* distance,int trainIdx, int testIdx){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(numberToCalc > idx){
        float* seriesA = test_data  + (length * testIdx);
        float* seriesB = train_data + (length * trainIdx);
        int k = 0;
        int start = numberToCalc - 1;
        if(current > length - 1){
            start = current + (length-1) * (length - numberToCalc);
        }
        int arrayIdx = start + idx * (length - 1);
        int i = arrayIdx / length;
        int j = arrayIdx % length;
        float dist;
        start = max(0,i - window);
        int end = min(length, i + window);
        if(j >= start && j < end){
            //printf("seriesA[%d] : %f seriesB[%d] : %f\n",i,seriesA[i],j, seriesB[j]);
            dist = pow(seriesA[i] - seriesB[j], 2);
            //printf("(%d,%d) - Distance %f\n",i,j, dist);
            float left = /*idx > 0 && j > 0*/ j > 0 ? distance[arrayIdx - 1] : INT_MAX;
            float top = arrayIdx - length >= 0 ? distance[arrayIdx - length] : INT_MAX;
            float diagonal = arrayIdx - length > 0 && j > 0 ? distance[arrayIdx - length - 1] : INT_MAX;
            
            diagonal = (i == 0 && j == 0) ? 0 : diagonal;
            diagonal = min(diagonal,left);
            dist = dist + min(top,diagonal);
            distance[arrayIdx] = dist;
            //printf("%d %d %f %f %f Min Value : %f Final : %f\n",i,j,
            //left,top,diagonal, min(top,diagonal), distance[arrayIdx]);
        }
    }
}


__global__ void tempCheck(float* A, float* B, int length)
{
    int i = 0;
    for(i = 0; i < length; i++){
        printf("%f %f\n",A[i],B[i]);
    }
}
__global__ void LB_Keogh(float* test_data, float* train_data, int length, 
    int window, int trainIdx, int testIdx, float* d_LB_dist){
    int i = threadIdx.x, start_idx = 0, end_idx, j;

    extern __shared__ float output[];
    if(i < length){
        float* seriesA = test_data  + (length * testIdx);
        float* seriesB = train_data + (length * trainIdx);
        float lower_bound = INT_MAX, upper_bound = INT_MIN, current;            
        lower_bound = INT_MAX;
        upper_bound = INT_MIN;
        start_idx = i - window >= 0 ? i - window : 0;
        end_idx = min(i + window, length);
        for(j = start_idx; j < end_idx; j++){
            if(seriesB[j] > upper_bound)
                upper_bound = seriesB[j];

            if(seriesB[j] < lower_bound)
                lower_bound = seriesB[j];
        }
        
        current = seriesA[i];
        float val = 0;
        if (current>upper_bound)
            val =  pow(current-upper_bound,2);
        else if (current<lower_bound)
            val =  pow(current-lower_bound,2);
        
        output[i] =val;

    	__syncthreads();

    	if (i == 0)
    	{
    		d_LB_dist[0] = 0;
    		int k;
    		for (k = 0; k < length; k++)
    		{
    			d_LB_dist[0] += output[k];
    		}
    		d_LB_dist[0] = sqrt(d_LB_dist[0]);
    	}
    }
}



////////////////////////////////////////////////////////////////
float DTWDistance_CPU(float* seriesA, float* seriesB, int length, int window){
    int i = -1, j = -1;
    float* distances = (float*) malloc(length * length * sizeof(float));
    float dist;
    //memset(distances,FLT_MAX ,(length) * (length));
    uint64_t k;
    for(k = 0; k < length * length; k++){
        distances[k] = FLT_MAX;
    }

    for(i = 0; i < length; i++){
        int start = max(0,i - window);
        int end = min(length, i + window);
        
        for(j = start; j < end; j++){
            //printf("seriesA[%d] : %f seriesB[%d] : %f\n",i,seriesA[i],j, seriesB[j]);
            dist = pow(seriesA[i] - seriesB[j], 2);
            //printf("(%d,%d) - Distance %f\n",i,j, dist);
            int idx = (i * length) + j;
            float left = /*idx > 0 && j > 0*/ j > start ? distances[idx - 1] : INT_MAX;
            float top = idx - length >= 0 ? distances[idx - length] : INT_MAX;
            float diagonal = idx - length > 0 && j > 0 ? distances[idx - length - 1] : INT_MAX;
            diagonal = (i == 0 && j == 0) ? 0 : diagonal;
            diagonal = min(diagonal,left);
            distances[idx] = dist + min(top,diagonal);
            //printf("%d %d %f %f %f Min Value : %f Final : %f\n",i,j,
            //left,top,diagonal, min(top,diagonal), distances[idx]);
        }
    }

    /*for(i = 0 ; i < length; i++){
        for(j = 0; j < length; j++){
            int idx = (i * length) + j;
            printf("%d %d %f | ",i,j, distances[idx]);
        }
        printf("\n");
    }*/
    float result = sqrt(distances[length * length - 1]);
    return result;
}


float LB_Keogh_CPU(float* seriesA, float* seriesB, int length, int window){
    float LB_sum = 0;
    int i = 0, start_idx = 0, end_idx, j;
    float lower_bound = INT_MAX, upper_bound = INT_MIN, current;
    
    for(;i < length; i++){
        
        lower_bound = INT_MAX;
        upper_bound = INT_MIN;
        start_idx = i - window >= 0 ? i - window : 0;
        end_idx = min(i + window, length);
        for(j = start_idx; j < end_idx; j++){
            if(seriesB[j] > upper_bound)
                upper_bound = seriesB[j];

            if(seriesB[j] < lower_bound)
                lower_bound = seriesB[j];
        }
        current = seriesA[i];
        //printf("%f %d %f %f %f\n", seriesA[i], i, lower_bound, upper_bound, LB_sum);
        float val = 0;
        if (current>upper_bound)
            val =  pow(current-upper_bound,2);
        else if (current<lower_bound)
            val =  pow(current-lower_bound,2);
        //printf("%d %f\n",i,val);
        LB_sum +=val;
         
    }
    return sqrt(LB_sum);
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
    int window_size = lenOfTimeSeries / 10;
    int LB_Keogh_param = 5;
    // If Testing File is provided
    if(argc > 5 || isDefault == 1){
        char* testFileName;
        if(isDefault == 0){
            testFileName = argv[4];
            noOfTestTimeSeries = atoi(argv[5]);
        }else{
            testFileName = "SonyAIBORobotSurface_TEST";
        }

        if(argc > 6){
            window_size = atoi(argv[6]);
        }
        if(argc > 7){
            LB_Keogh_param = atoi(argv[7]);
        }
        printf("-----------------------------------------------------\n");
        //get testing file
        printf("Reading test file\n");
        test_size = noOfTestTimeSeries * lenOfTimeSeries * sizeof(float);
        test_data = (float*) malloc (test_size);
        test_class = (int *) malloc(noOfTestTimeSeries * sizeof(int));
        //read test file
        readfile(testFileName, test_data, test_class, noOfTestTimeSeries);
        
        printf("Testing File : %s\n",testFileName);
        printf("Number of Time Series to validate: %d\n",noOfTestTimeSeries);
        printf("Window Size for kNN: %d\n",window_size);
        printf("LB Keogh Parameter: %d\n",LB_Keogh_param);
    }


    
    printf("===================================================\n");

    //GPU number present in the system
    //int noOfGPUs;
    //checkCudaErrors(cudaGetDeviceCount(&noOfGPUs));
    //printf("Total GPUs on System : %d\n", noOfGPUs);
    //checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceReset());
    cudaProfilerStart();
    int i = 0, j = 0, errorCount = 0;
    start = clock();
    
    float* d_test_data;
    clock_t timer = clock();
    checkCudaErrors(cudaMalloc((void**)&d_test_data, test_size));
    checkCudaErrors(cudaMemcpy(d_test_data, test_data, test_size, cudaMemcpyHostToDevice));
    printf("Time to upload test data %f\n", 
    (double)(clock() - timer) / ((double)CLOCKS_PER_SEC));

    // Transfer train data
    float* d_train_data = 0;
    timer = clock();
    checkCudaErrors(cudaMalloc((void**)&d_train_data, train_size));
    checkCudaErrors(cudaMemcpy(d_train_data, train_data, train_size, cudaMemcpyHostToDevice));
    printf("Time to upload train data %f\n", 
    (double)(clock() - timer) / ((double)CLOCKS_PER_SEC));
    int k;

    //float* d_norm_train_data = 0;
    //checkCudaErrors(cudaMalloc((void**)&d_norm_train_data, train_size));

    // Normalize Train Series
    int threadsPerBlock = min((int)ceil(noOfTimeSeries/(float)32)*32,1024);
    int noOfBlocks = ceil((noOfTimeSeries)/(float)threadsPerBlock);
    //NormalizeTimeSeries<<<noOfBlocks, threadsPerBlock>>>(d_train_data, noOfTimeSeries, lenOfTimeSeries,d_norm_train_data);
    //checkCudaErrors(cudaDeviceSynchronize());
    //checkCudaErrors(cudaMemcpy(d_train_data, d_norm_train_data, train_size, cudaMemcpyDeviceToDevice));

    float* d_LB_dist;
    float min_dist, dist;
    //printf("Normalized Train Series\n");
    //tempCheck<<<1,1>>>(d_norm_train_data,d_train_data,noOfTimeSeries * lenOfTimeSeries);
    int predictedIndex = 0;
    for(i = 0; i < noOfTestTimeSeries; i++){
        min_dist = FLT_MAX;
        for(j = 0; j < noOfTimeSeries; j++){
            
            float* LB_dist=(float*)malloc(sizeof(float));
	    
            checkCudaErrors(cudaMalloc((void**)&d_LB_dist, sizeof(float)));
            
            threadsPerBlock = min((int)ceil(lenOfTimeSeries/(float)32)*32,1024);
            LB_Keogh<<<1,threadsPerBlock,lenOfTimeSeries*sizeof(float)>>>(d_test_data,d_train_data,lenOfTimeSeries,LB_Keogh_param,j,i,d_LB_dist);
	        checkCudaErrors(cudaMemcpy(LB_dist, d_LB_dist, sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(d_LB_dist));
            //printf("LB_Keogh between %d and %d is %f\n",i,j,LB_dist[0]);
            float* test = test_data  + (lenOfTimeSeries * i);
            float* train = train_data + (lenOfTimeSeries * j);
            if(LB_dist[0] < min_dist){
                float* distances = (float*) malloc(lenOfTimeSeries * lenOfTimeSeries * sizeof(float));
                float* d_distances;
                checkCudaErrors(cudaMalloc((void**)&d_distances, lenOfTimeSeries * lenOfTimeSeries * sizeof(float)));
                // init to FLT_MAX
                checkCudaErrors(cudaMemset(d_distances,FLT_MAX, lenOfTimeSeries * lenOfTimeSeries * sizeof(float)));
                for(k = 0; k < (2 * lenOfTimeSeries - 1); k++){

                    if(k < lenOfTimeSeries){
                        threadsPerBlock = k + 1;
                    }else{
                        threadsPerBlock = (2 * lenOfTimeSeries) - k - 1;
                    }
                    int temp = threadsPerBlock;
                    threadsPerBlock = min((int)ceil(threadsPerBlock/(float)32)*32,1024);
                    noOfBlocks = ceil((temp)/(float)threadsPerBlock);
                    DTWDistance<<<noOfBlocks,threadsPerBlock>>>(d_test_data,d_train_data,lenOfTimeSeries, window_size,temp,k,d_distances, j,i);
                    cudaThreadSynchronize();
                }
                checkCudaErrors(cudaMemcpy(distances, d_distances,lenOfTimeSeries * lenOfTimeSeries * sizeof(float), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaFree(d_distances));

                dist = distances[lenOfTimeSeries * lenOfTimeSeries - 1];
                dist = sqrt(dist);
                free(distances);
                
                if(dist < min_dist){
                    min_dist = dist;
                    predictedIndex = j;
                    //printf("Min Distance between %d and %d is %f\n",i,j,dist);
                }
            }
            
        }

        // Check for stuff
        int predictedClass = train_class[predictedIndex];
        int correctClass = test_class[i];
        //printf("Test Case : %d Predicted Class :%d Actual Class : %d\n",i, predictedClass, correctClass);
        if(predictedClass != correctClass){
            errorCount++;
        }
    }
    end = clock() - start;
    cudaProfilerStop();
    checkCudaErrors(cudaDeviceReset());
    cudaFree(d_LB_dist);
    cudaFree(d_train_data);
    cudaFree(d_test_data);
    
    double endtime = (double)end / ((double)CLOCKS_PER_SEC);
    printf("Total GPU Time %f\n", endtime);
    printf("GPU Accuracy is %f\n",(float)(noOfTestTimeSeries-errorCount)*(100.0/noOfTestTimeSeries));    
    //printf("--------------------------------------------\n");
    //printf("CPU : \n");
    errorCount = 0;
    start = clock();
    for(i = 0; i < noOfTestTimeSeries; i++){
        min_dist = FLT_MAX;
        for(j = 0; j < noOfTimeSeries; j++){
            float* test = test_data  + (lenOfTimeSeries * i);
            float* train = train_data + (lenOfTimeSeries * j);
            float LB_dist = LB_Keogh_CPU(test,train,lenOfTimeSeries,LB_Keogh_param);
            //printf("LB_Keogh between %d and %d is %f\n",i,j,LB_dist);
            if(LB_dist < min_dist){
                dist = DTWDistance_CPU(test,train,lenOfTimeSeries, window_size);
                if(dist < min_dist){
                    min_dist = dist;
                    predictedIndex = j;
                    //printf("Min Distance between %d and %d is %f\n",i,j,dist);
                }
            }
        }
        // Check for stuff
        int predictedClass = train_class[predictedIndex];
        int correctClass = test_class[i];
        //printf("Test Case : %d Predicted Class :%d Actual Class : %d\n",i, predictedClass, correctClass);
        if(predictedClass != correctClass){
            errorCount++;
        }
    }
    free(train_class);
    free(test_class);
    
    
    end = clock() - start;
    endtime = (double)end / ((double)CLOCKS_PER_SEC);
    //printf("Total CPU Time : %f\n", endtime);
    //printf("CPU Accuracy : %f\n",(float)(noOfTestTimeSeries-errorCount)*(100.0/noOfTestTimeSeries));    
    return 0;

}
