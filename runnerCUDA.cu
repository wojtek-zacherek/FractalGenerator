#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "stb_image_write.h"


#define DEBUG 0

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

#define someNum1 127+128*3
__device__ volatile complex var1 = {.real = 2, .imag = 0.25}, 
    var2 = {.real = .2, .imag = .33};
__device__ volatile int foiterationsMaxund = 0;
__device__ volatile int iterationsMax = someNum1;
__device__ volatile double thresh = 2;
int iterationsMaxHost = someNum1;

struct timespec start, finish;
double elapsed;
#define xResDEF 15360
#define yResDEF 8640
// #define xResDEF 1920
// #define yResDEF 1080
#define myTestX 29
#define myTestY 4
#define DEBUG_ID (myTestX+1)*(myTestY+1)-1
__device__ volatile uint yResD = yResDEF;
__device__ volatile uint xResD = xResDEF;
uint yRes = yResDEF;
uint xRes = xResDEF;

int doSomething(uint, uint, double, double, double, double ,double, uint);
void makeColourfull(char**, double*, uint, uint, uint);

__global__ void doMath(double *a, double *b, double *c, int n){

    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;

    unsigned int idx=ix+iy*n;
    // printf("%d - %d - %d\n",threadIdx.x, blockIdx.x, idx);

    // c[threadIdx.x*yResD + blockIdx.x] = idx;
    // c[idx] = idx;

    int myX = ix/(xResD+1);
    int myY = ix - (ix/(xResD+1)*(xResD+1));
    if (idx < (xResD+1)*(yResD+1)){
        complex Zn = {.real = 0, .imag = 0},
            // C = {.real = b[myX], .imag = a[myY]};
            C = {.real = a[myY], .imag = b[myX]};
        
        uint numberOfIterations = 0;
        double magnitude = sqrt(pow((&Zn)->real,2) + pow((&Zn)->imag,2));
        
        if(idx == DEBUG_ID){
            printf("Starting: C=%f+%fi\n",(&C)->real,(&C)->imag);
            // printf("Starting: C=%f+%fi\n",a[0],b[0]);
            printf("Magnitude: Zn=%f+%fi -> %f\n",(&Zn)->real,(&Zn)->imag, magnitude);
            printf("Indicies: %d,%d   Vals: %4f + %4fi\n",myX, myY, (&C)->real,(&C)->imag);
        }
        

        while(magnitude <= (uint)thresh && numberOfIterations < iterationsMax){
            if(idx == DEBUG_ID){
                printf("Iter %d ------------------------------------\n",numberOfIterations);
            }

            numberOfIterations++;

            // complexAdd(&Zn,&C); 
            (&Zn)->real += (&C)->real;
            (&Zn)->imag += (&C)->imag;
            if(idx == DEBUG_ID){
                printf("Complex Add: Zn=%f+%fi\n",(&Zn)->real,(&Zn)->imag);
            }
            
            // complexPower(&Zn,2)
            double r = sqrt(pow((&Zn)->real,2) + pow((&Zn)->imag,2));
            double theta;
            if((&Zn)->real == 0){
                if((&Zn)->imag >= 0){
                    theta = 3.1415/2;
                }else{
                    theta = -3.1415/2;
                }
            }else{
                theta = atan((&Zn)->imag/(&Zn)->real);
            }
            (&Zn)->real = pow(r,2)*cos(2*theta);
            (&Zn)->imag = pow(r,2)*sin(2*theta);
            // double x = (&Zn)->real * (&Zn)->real - (&Zn)->imag * (&Zn)->imag;
            // double y = (&Zn)->real * (&Zn)->imag + (&Zn)->imag * (&Zn)->real;
            // (&Zn)->real = x;
            // (&Zn)->imag = y;
            if(idx == DEBUG_ID){
                printf("Complex Power: Zn=%f+%fi: r=%f, theta=%f\n",(&Zn)->real,(&Zn)->imag,r,theta);
                // printf("Complex Power: Zn=%f+%fi\n",(&Zn)->real,(&Zn)->imag);
            }

            // // complexAdd(&Zn,&var2); 
            // (&Zn)->real += (&var2)->real;
            // (&Zn)->imag += (&var2)->imag;
            // if(idx == DEBUG_ID){
            //     printf("Complex Add: Zn=%f+%fi\n",(&Zn)->real,(&Zn)->imag);
            // }

            

            // complexMult(&Zn,&var2);
            // double x = (&Zn)->real * (&var2)->real - (&Zn)->imag * (&var2)->imag;
            // double y = (&Zn)->real * (&var2)->imag + (&Zn)->imag * (&var2)->real;
            // (&Zn)->real = x;
            // (&Zn)->imag = y;
            // if(idx == DEBUG_ID){
            //     printf("Complex Mult: Zn=%f+%fi\n",(&Zn)->real,(&Zn)->imag);
            // }

            magnitude = sqrt(pow((&Zn)->real,2) + pow((&Zn)->imag,2));
            if(idx == DEBUG_ID){
                printf("Magnitude: Zn=%f+%fi -> %f\n",(&Zn)->real,(&Zn)->imag, magnitude);
            }
        }
        if(idx <= 0){
            printf("%d - %d - %d\n",threadIdx.x, blockIdx.x, numberOfIterations-1);
        }
        // c[threadIdx.x*yResD + blockIdx.x] = numberOfIterations-1;
        // c[threadIdx.x*yResD + blockIdx.x] = threadIdx.x*yResD + blockIdx.x;
        c[idx] = numberOfIterations-1;
        // c[idx] = idx;
        // c[idx] = a[myY];
        // c[idx] = b[0];
        // c[idx] = ix - (ix/(xResD+1)*(xResD+1));
    }
}


int main( int argc, char* argv[] ){
    // uint yRes = 500;
    // uint xRes = 889;
    
    double thresh = 1;
    uint iter = 127+128;
    // double xMin = -0.45;
    // double xMax = -0.15;
    // double yMin = 0.4;
    // double yMax = .6;
    // double xMin = 0;
    // double xMax = 0.35;
    // double yMin = 0.65;
    // double yMax = .9;
    double xMin = -2;
    double xMax = 1;
    double yMin = -1;
    double yMax = 1;
    // double ratio = 1.5;


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
    double *h_c;
 
    // Device input vectors
    double *d_x;
    double *d_y;
    //Device output vector
    double *d_c;

    uint resOffset = 1;

    //
    uint xBytes = (xResolution + resOffset) * sizeof(double);
    uint yBytes = (yResolution + resOffset) * sizeof(double);
    uint cBytes = ((xResolution + resOffset) * (yResolution + resOffset)) * sizeof(double);

    // Allocate memory for each vector on host
    h_x = (double*)malloc(xBytes);
    h_y = (double*)malloc(yBytes);
    h_c = (double*)malloc(cBytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_x, xBytes);
    cudaMalloc(&d_y, yBytes);
    cudaMalloc(&d_c, cBytes);


    
    double xSlope = (double)(xMax - xMin) / xResolution;
    double ySlope = (double)(yMax - yMin) / yResolution;
    if(DEBUG == 1){
        printf("Slopes: %f %f\n",xSlope, ySlope);
    }
    

    // Indicies
    int xStart = 0;
    int xEnd = xResolution + resOffset;
    int yStart = 0;
    int yEnd = yResolution + resOffset;

    

    // double tempX, tempY;
    printf("h_x\n");
    for(int l = xStart; l < xEnd; l++){
        h_x[l] = xSlope*((double)l - 0) + xMin;
        if(DEBUG == 1){
            printf("%d : %f\n",l,h_x[l]);
        }
    }
    printf("h_y\n");
    for(int l = yStart; l < yEnd; l++){
        h_y[l] = ySlope*((double)l - 0) + yMin;
        if(DEBUG == 1){
            printf("%d : %f\n",l,h_y[l]);
        }
    }
    printf("%f...%f\n",h_y[0],h_y[yEnd-1]);

    // Copy host vectors to device
    cudaMemcpy( d_x, h_x, xBytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_y, h_y, yBytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, cBytes, cudaMemcpyHostToDevice);
    //

    // int blockSize, gridSize;
    // Multiple of 32.
    dim3 gridSize((xResolution/1024.0*yResolution + 1),1);
    dim3 blockSize(1024,1);
    // Number of threads in each thread block
    // blockSize = xResolution;
    // Number of thread blocks in grid
    // gridSize = (int)ceil(((float)n)/blockSize);        
 
    // printf("blockSize = %d, gridSize = %d\n",blockSize,gridSize);

    // Execute the kernel
    doMath<<<gridSize, blockSize>>>(d_x, d_y, d_c, n);

    // Copy array back to host
    cudaMemcpy( h_c, d_c, cBytes, cudaMemcpyDeviceToHost );
        
    // uint val = doMath(tempX, tempY, thresh, iter);


    printf("Hello\n");
    // for(uint i = 0; i < xResolution; i++){
    //     // printf("what\n");
    //     for(uint j = 0; j < yResolution; j++){
    //         // printf("%4d", h_c[i*yResolution + j]);
    //         // printf("%4d",h_c[j * (xResolution + resOffset) + i]);
    //     }
    //     if(DEBUG == 1){
    //         // printf("\n");
    //     }
    // }
    printf("Tested Indicies: %d,%d    Vals: %4f + %4fi\n",myTestX,myTestY,h_x[myTestX],h_y[myTestY]);

    // for(uint i = 0; i < xResolution + resOffset; i++){
    //     for(uint j = 0; j < yResolution + resOffset; j++){
    //         if(DEBUG == 1){
    //             printf("%4d",h_c[ i*(yResolution+resOffset) + j]);
    //         }
    //     }
    //     if(DEBUG == 1){
    //         printf("\n");
    //     }
    // }


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
    
    // pgmimg = fopen(filename, "a");
    // free(filename);
    
    
    // fprintf(pgmimg, "P2\n");
    // fprintf(pgmimg, "%d %d\n",xResolution + resOffset,yResolution + resOffset);
    // fprintf(pgmimg, "%d\n",iter);
    


    // for(uint j = 0; j < yResolution + resOffset; j++){
    //     for(uint i = 0; i < xResolution + resOffset; i++){
    //         fprintf(pgmimg, "%d ",(int)h_c[j * (xResolution + resOffset) + i]);
    //         if(DEBUG == 1){
    //             printf("%4d",(int)h_c[j * (xResolution + resOffset) + i]);
    //         }
    //     }
    //     if(DEBUG == 1){
    //         printf("\n");
    //     }
    //     fprintf(pgmimg, "\n");
    // }
    // fclose(pgmimg);



    char *data;
    makeColourfull(&data, h_c, xResolution, yResolution, iter);
    printf("Value: %d %d %d\n",data[0],data[xResolution*yResolution/2 + xResolution/2],data[xResolution*yResolution]);
    // stbi_write_jpg(filename2,xResolution,yResolution,1,data,100);
    stbi_write_jpg(filename2,xResolution,yResolution,3,data,90);
    // stbi_write_png(filename3,xResolution,yResolution,3,data,0);
    free(data);

    
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



void makeColourfull(char **target, double *matrix, uint xResolution, uint yResolution, uint iter){
    int numColors = iter;
    char maxRGBValue = 255;
    struct rgb colors[numColors];
    struct rgb redwhite[512];
    struct rgb blue[256];
    struct rgb bluewhite[256];
    struct rgb blackredorangewhite[256+128+128];
    colorSelection colorSelected;

    colorSelected.selection = redwhite; colorSelected.numEntries = 511;
    colorSelected.selection = colors; colorSelected.numEntries = iter;
    colorSelected.selection = bluewhite; colorSelected.numEntries = 255;
    colorSelected.selection = blue; colorSelected.numEntries = 255;
    colorSelected.selection = blackredorangewhite; colorSelected.numEntries = 511;

    for(int i = 0; i < 256; i++){
        blue[i].r = 0;
        blue[i].g = 0;
        blue[i].b = i;
    }

    for(int i = 0; i < 256; i++){
        blackredorangewhite[i].r = i;
        blackredorangewhite[i].g = 0;
        blackredorangewhite[i].b = 0;
    }
    for(int i = 0; i < 128; i++){
        blackredorangewhite[i + 256].r = 255;
        blackredorangewhite[i + 256].g = i;
        blackredorangewhite[i + 256].b = 0;
    }
    for(int i = 0; i < 128; i++){
        // blackredorangewhite[i + 256 + 128].r = 255;
        // blackredorangewhite[i + 256 + 128].g = 128 + i;
        // blackredorangewhite[i + 256 + 128].b = 2*i;
        blackredorangewhite[i + 256 + 128].r = 255 - 2*i;
        blackredorangewhite[i + 256 + 128].g = 128 - i;
        blackredorangewhite[i + 256 + 128].b = 0;
    }

    for(int i = 0; i < 128; i++){
        bluewhite[i].r = 0;
        bluewhite[i].g = 0;
        bluewhite[i].b = 2*i;
    }
    for(int i = 0; i < 128; i++){
        bluewhite[i + 128].r = 2*i;
        bluewhite[i + 128].g = 2*i;
        bluewhite[i + 128].b = 255;
    }

    for(int i = 0; i < numColors; i++){
        colors[i].r = i;
        colors[i].g = 0;
        colors[i].b = 0;
    }


    for(int i = 0; i < 256; i++){
        redwhite[i].r = i;
        redwhite[i].g = 0;
        redwhite[i].b = 0;
    }
    for(int i = 0; i < 256; i++){
        redwhite[i + 256].r = 255;
        redwhite[i + 256].g = i;
        redwhite[i + 256].b = i;
    }

    // ratio = colorSelected.numEntries/maxValue
    printf("max = %d\n",iterationsMaxHost);
    
    *target = (char*)malloc(xResolution*yResolution*sizeof(char)*3);
    char *data = *target;
    uint index = 0;
    uint index2 = 0;
    for(uint i = 0; i < xResolution; i++){
        for(uint j = 0; j < yResolution; j++){
            index = (int)(((long)matrix[i + j*(xResolution+1)] * colorSelected.numEntries) / iterationsMaxHost);
            index2 = 3*(i + j*xResolution);
            
            data[index2 + 0] = colorSelected.selection[index].r;
            data[index2 + 1] = colorSelected.selection[index].g;
            data[index2 + 2] = colorSelected.selection[index].b;
        }
    }
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