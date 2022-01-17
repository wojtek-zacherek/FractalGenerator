#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "stb_image_write.h"


#define DEBUG 1

typedef struct complex {
    double real;
    double imag;
} complex;

typedef struct rgb{
    char r;
    char g;
    char b;
} rgb;

typedef struct colorSelection {
    struct rgb *selection;
    uint numEntries;
} colorSelection;

__device__ volatile complex var1 = {.real = 2, .imag = 0.25}, 
    var2 = {.real = 2, .imag = 2};
__device__ volatile int foiterationsMaxund = 0;
__device__ volatile int iterationsMax = 127;
__device__ volatile double thresh = 1;

struct timespec start, finish;
double elapsed;

int doSomething(uint, uint, double, double, double, double ,double, uint);


__global__ void doMath(double *a, double *b, int *c, int n){

    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;

    unsigned int idx=ix+iy*n;
    printf("%d - %d - %d\n",threadIdx.x, blockIdx.x, idx);

    if (threadIdx.x < 15 && blockIdx.x < 10){
        complex Zn = {.real = 0, .imag = 0},
            C = {.real = a[threadIdx.x], .imag = b[blockIdx.x]};
        
        uint numberOfIterations = 0;
        uint magnitude = sqrt(pow((&Zn)->real,2) + pow((&Zn)->imag,2));
        

        while(magnitude <= (uint)thresh && numberOfIterations < iterationsMax){
            numberOfIterations++;
            // complexAdd(&Zn,&C); 
            (&Zn)->real += (&C)->real;
            (&Zn)->imag += (&C)->imag;
            
            // complexPower(&Zn,2)
            double r = sqrt(pow((&Zn)->real,2) + pow((&Zn)->imag,2));
            double theta = atan((&Zn)->imag/(&Zn)->real);
            (&Zn)->real = pow(r,2)*cos(2*theta);
            (&Zn)->imag = pow(r,2)*sin(2*theta);

            // complexMult(&Zn,&var2);
            int x = (&Zn)->real * (&var2)->real - (&Zn)->imag * (&var2)->imag;
            int y = (&Zn)->real * (&var2)->imag + (&Zn)->imag * (&var2)->real;
            (&Zn)->real = x;
            (&Zn)->imag = y;
        }
        
        printf("%d - %d - %d\n",threadIdx.x, blockIdx.x, numberOfIterations-1);
        c[threadIdx.x*10 + blockIdx.x] = numberOfIterations-1;
    }
}


int main( int argc, char* argv[] ){
    // uint yRes = 500;
    // uint xRes = 889;
    uint yRes = 20;
    uint xRes = 25;
    double thresh = 1;
    uint iter = 127;
    double xMin = -2;
    double xMax = 1;
    double yMin = -1;
    double yMax = 1;
    double ratio = 1.5;


    // if(argc == 3){
    //     xRes = atoi(argv[1]);
    //     yRes = atoi(argv[2]);
    // }else if(argc == 4){
    //     xRes = atoi(argv[1]);
    //     yRes = atoi(argv[2]);
    //     thresh = atoi(argv[3]);
    // }else if(argc == 5){
    //     xRes = atoi(argv[1]);
    //     yRes = atoi(argv[2]);
    //     thresh = atoi(argv[3]);
    //     iter = atoi(argv[4]);
    // }else if(argc == 6){
    //     xRes = atoi(argv[1]);
    //     yRes = atoi(argv[2]);
    //     thresh = atoi(argv[3]);
    //     iter = atoi(argv[4]);
    //     if(atoi(argv[5]) == 0){
    //         yMin = -1;
    //         yMax = 1;
    //     }else{
    //         xMin = ((xMax +xMin)/2.0) - ratio*(yMax - yMin)/2.0;
    //         xMax = ((xMax +xMin)/2.0) + ratio*(yMax - yMin)/2.0;
    //     }
    // }

    printf("Computing for %d x %d, %d iterations, threshold of %g, ranging on x-axis %g..%g and y-axis %g..%g\n",xRes,yRes,iter,thresh,xMin,xMax,yMin,yMax);
    clock_gettime(CLOCK_MONOTONIC, &start);
    doSomething(xRes, yRes, xMin, xMax, yMin, yMax, thresh, iter);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time taken %f \n", elapsed);

    return 1;


}

int doSomething(uint xResolution, uint yResolution, double xMin, double xMax, double yMin, double yMax, double thresh, uint iter){
    // Size of vectors
    int n = xResolution*yResolution;
 
    // Host input vectors
    double *h_x;
    double *h_y;
    //Host output vector
    int *h_c;
 
    // Device input vectors
    double *d_x;
    double *d_y;
    //Device output vector
    int *d_c;

    uint resOffset = 1;

    //
    uint xBytes = (xResolution + resOffset) * sizeof(double);
    uint yBytes = (yResolution + resOffset) * sizeof(double);
    uint cBytes = ((xResolution + resOffset) * (yResolution + resOffset)) * sizeof(int);

    // Allocate memory for each vector on host
    h_x = (double*)malloc(xBytes);
    h_y = (double*)malloc(yBytes);
    h_c = (int*)malloc(cBytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_x, xBytes);
    cudaMalloc(&d_y, yBytes);
    cudaMalloc(&d_c, cBytes);


    
    double xSlope = (double)(xMax - xMin) / xResolution;
    double ySlope = (double)(yMax - yMin) / yResolution;
    if(DEBUG == 1){
        printf("Slopes: %f %f\n",xSlope, ySlope);
    }
    
    int xDivs = 1;
    int yDivs = 1;
    int xDivStep = (xResolution + resOffset) / xDivs;
    int yDivStep = (yResolution + resOffset) / yDivs;


    uint xStart = (xDivs - (0 + 1)) * xDivStep;
    uint xEnd = xStart + xDivStep;
    uint yStart = (yDivs - (0 + 1)) * yDivStep;
    uint yEnd = yStart + yDivStep;


    // double tempX, tempY;
    for(uint l = xStart; l < xEnd; l++){
        h_x[l] = xSlope*((double)l - 0) + xMin;
        if(DEBUG == 1){
            printf("%d : %f\n",l,h_x[l]);
        }
    }
    for(uint l = yStart; l < yEnd; l++){
        h_y[l] = ySlope*((double)l - 0) + yMin;
        if(DEBUG == 1){
            printf("%d : %f\n",l,h_y[l]);
        }
    }


    // Copy host vectors to device
    cudaMemcpy( d_x, h_x, xBytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_y, h_y, yBytes, cudaMemcpyHostToDevice);

    // int blockSize, gridSize;
    dim3 gridSize(yResolution,1);
    dim3 blockSize(xResolution,1);
    // Number of threads in each thread block
    // blockSize = xResolution;
    // Number of thread blocks in grid
    // gridSize = (int)ceil(((float)n)/blockSize);        
 
    // printf("blockSize = %d, gridSize = %d\n",blockSize,gridSize);

    // Execute the kernel
    doMath<<<gridSize, blockSize>>>(d_x, d_x, d_c, n);

    // Copy array back to host
    cudaMemcpy( h_c, d_c, cBytes, cudaMemcpyDeviceToHost );
        
    // uint val = doMath(tempX, tempY, thresh, iter);


    printf("Hello\n");
    for(uint i = 0; i < xResolution; i++){
        printf("what\n");
        for(uint j = 0; j < yResolution; j++){
            // printf("%d,%d : %d\n", i, j, h_c[i*yResolution + j]);
        }
    }
    printf("Hello\n");


    // matrix[i][j] = val;
    // if(maxValue < matrix[i][j]){
    //     maxValue = matrix[i][j];
    // }
    



    FILE* pgmimg;
    // pgmimg = fopen("pgmimg.pgm", "wb");
    size_t fileTime = time(NULL);
    size_t fileSize = snprintf(NULL, 0, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime) + 1;
    size_t fileSize2 = snprintf(NULL, 0, "%d_%d_%g_%lu.jpeg", xResolution, yResolution, thresh, fileTime) + 1;
    size_t fileSize3 = snprintf(NULL, 0, "%d_%d_%g_%lu.png", xResolution, yResolution, thresh, fileTime) + 1;
    char* filename = (char *)malloc(fileSize);
    char* filename2 = (char *)malloc(fileSize2);
    char* filename3 = (char *)malloc(fileSize3);
    snprintf(filename, fileSize, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime);
    snprintf(filename2, fileSize2, "%d_%d_%g_%lu.jpeg", xResolution, yResolution, thresh, fileTime);
    snprintf(filename3, fileSize3, "%d_%d_%g_%lu.png", xResolution, yResolution, thresh, fileTime);
    
    pgmimg = fopen(filename, "a");
    free(filename);
    
    
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n",xResolution + resOffset,yResolution + resOffset);
    fprintf(pgmimg, "%d\n",iter);
    

    for(uint j = 0; j < yResolution + resOffset; j++){
        for(uint i = 0; i < xResolution + resOffset; i++){
            fprintf(pgmimg, "%d ",h_c[i * (xResolution + resOffset) + j]);
            if(DEBUG == 1){
                printf("%4d",h_c[i * (xResolution + resOffset) + j]);
            }
        }
        if(DEBUG == 1){
            printf("\n");
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);






    
    // Release device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_c);
 
    // Release host memory
    free(h_x);
    free(h_y);
    free(h_c);

    return 0;
    

}












// #include <cuda_runtime.h>
// #include <stdio.h>



// void initialint(int *ip,int size)
// {
//     for(int i=0;i<size;i++)
//         ip[i]=i;

// }

// void printmatrix(int *C,const int nx,const int ny)
// {
//     int *ic=C;
//     printf("\n Matrix: (%d.%d) \n",nx,ny);
//     for(int i=0;i<ny;i++){
//         for(int j=0;j<nx;j++){
//             printf("%3d",ic[j+nx*i]);}
//     printf("\n");

//     }
// printf("\n");
// }

// __global__ void printthreadindex(int *A,const int nx,const int ny)
// {
//     int ix=threadIdx.x+blockIdx.x*blockDim.x;
//     int iy=threadIdx.y+blockIdx.y*blockDim.y;

//     unsigned int idx=ix+iy*nx;

//     printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d  ival %2d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);

// }

// int main()
// {
//     int nx=8,ny=6;
//     int nxy=nx*ny;
//     int nBytes=nxy*sizeof(float);

//     int *h_A;
//     h_A=(int *)malloc(nBytes);

//     initialint(h_A,nxy);
//     printmatrix(h_A,nx,ny);

//     int *d_MatA;
//     cudaMalloc((void **)&d_MatA,nBytes);

//     cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
//     dim3 block(8,1);
//     dim3 grid(1,6);
//     printthreadindex <<<grid,block>>> (d_MatA,nx,ny);

//     cudaFree(d_MatA);
//     free(h_A);

//     system("pause");
//     return 0;



// }