#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include "stb_image_write.h"

struct timespec start, finish;
double elapsed;

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

void doSomething(uint, uint, double, double, double, double ,double, uint);
void makeColourfull(char**, uint**, uint, uint, uint);
uint doMath(double, double,double, uint);
uint complexMag(complex*);
void complexPower(complex*, uint);
void complexAdd(complex*, complex*);
void complexMult(complex*, complex*);

uint debug = 0;

int main(int argc, char *argv[]){
    // ./run.bin 1920 1080 8 255 1
    // !!! Toggle print statement debugging on(1)/off(0). Default is off
    debug = 0;
    uint yRes = 500;
    uint xRes = 889;
    double thresh = 1;
    uint iter = 127;
    double xMin = -0.3;
    double xMax = -0.15;
    double yMin = 0.425;
    double yMax = 0.5;
    double ratio = 1.5;
    if(argc == 3){
        xRes = atoi(argv[1]);
        yRes = atoi(argv[2]);
    }else if(argc == 4){
        xRes = atoi(argv[1]);
        yRes = atoi(argv[2]);
        thresh = atoi(argv[3]);
    }else if(argc == 5){
        xRes = atoi(argv[1]);
        yRes = atoi(argv[2]);
        thresh = atoi(argv[3]);
        iter = atoi(argv[4]);
    }else if(argc == 6){
        xRes = atoi(argv[1]);
        yRes = atoi(argv[2]);
        thresh = atoi(argv[3]);
        iter = atoi(argv[4]);
        if(atoi(argv[5]) == 0){
            yMin = -1;
            yMax = 1;
        }else{
            xMin = ((xMax +xMin)/2.0) - ratio*(yMax - yMin)/2.0;
            xMax = ((xMax +xMin)/2.0) + ratio*(yMax - yMin)/2.0;
        }
    }
    
    printf("Computing for %d x %d, %d iterations, threshold of %g, ranging on x-axis %g..%g and y-axis %g..%g\n",xRes,yRes,iter,thresh,xMin,xMax,yMin,yMax);

    clock_gettime(CLOCK_MONOTONIC, &start);
    doSomething(xRes, yRes, xMin, xMax, yMin, yMax, thresh, iter);
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time taken %f \n", elapsed);

    return 1;

}

uint maxValue = 0;
sem_t mutex;

void *myThreadFun(void *args)
{
    uint xStart =   *((uint *)(((void **)args)[0]));
    uint xEnd =     *((uint *)(((void **)args)[1]));
    uint yStart =   *((uint *)(((void **)args)[2]));
    uint yEnd =     *((uint *)(((void **)args)[3]));
    double xMin =     *((double *)(((void **)args)[4]));
    double yMin =     *((double *)(((void **)args)[5]));
    double thresh =   *((double *)(((void **)args)[6]));
    uint iter =     *((uint *)(((void **)args)[7]));
    double ySlope =   *((double *)(((void **)args)[8]));
    double xSlope =   *((double *)(((void **)args)[9]));
    uint **matrix = ((uint **)(((void **)args)[10])); 

    double tempX, tempY;
    for(uint i = xStart; i < xEnd; i++){
        tempX = xSlope*((double)i - 0) + xMin;
        for(uint j = yStart; j < yEnd; j++){
            tempY = ySlope*(j - 0) + yMin;
            if(debug == 1){
                printf("%d %d : %f %f\n",i,j,tempX,tempY);
            }
            uint val = doMath(tempX, tempY, thresh, iter);
            matrix[i][j] = val;
            // sem_wait(&mutex);
            if(maxValue < matrix[i][j]){
                maxValue = matrix[i][j];
            }
            // sem_post(&mutex);
        }
    }

    return NULL;
}

void doSomething(uint xResolution, uint yResolution, double xMin, double xMax, double yMin, double yMax, double thresh, uint iter){
    // Add 1 to the xRes and yRes when creating a matrix to include the endpoint(s). Also, this affects how the code below is made, so I created a variable
    // instead of hardcoding the value throughout.
    uint **matrix;
    
    uint resOffset = 1;
    matrix=malloc((xResolution + resOffset) * sizeof(uint*));
    // Check for NULL
    for(uint i = 0; i < xResolution + resOffset; i++){
        matrix[i]=malloc((yResolution + resOffset) * sizeof(uint));
        //again,check for NULL
    }

    
    double xSlope = (double)(xMax - xMin) / xResolution;
    double ySlope = (double)(yMax - yMin) / yResolution;
    if(debug == 1){
        printf("Slopes: %f %f\n",xSlope, ySlope);
    }
    
    int xDivs = 8;
    int yDivs = 8;
    int xDivStep = (xResolution + resOffset) / xDivs;
    int yDivStep = (yResolution + resOffset) / yDivs;
    void **argsArr = malloc(xDivs*yDivs*sizeof(void *));
    pthread_t *pthreadIDs = malloc(xDivs*yDivs*sizeof(pthread_t));
    uint **positions = malloc(5*xDivs*yDivs*sizeof(uint));
    // Check for NULL
    for(uint i = 0; i < xDivs*yDivs; i++){
        positions[i]=malloc(4 * sizeof(uint));
        //again,check for NULL
    }
    sem_init(&mutex, 0, 1);
    printf("Starting\n");
    for(int i  = 0; i < xDivs; i++){
        for(int j = 0; j < yDivs; j++){
            
            int k = i*xDivs + j;
            argsArr[k] = malloc(12*sizeof(void *) + sizeof(void **));
            
            uint xStart = (xDivs - (i + 1)) * xDivStep;
            uint xEnd = xStart + xDivStep;
            uint yStart = (yDivs - (j + 1)) * yDivStep;
            uint yEnd = yStart + yDivStep;
            positions[k][0] = xStart;
            positions[k][1] = xEnd;
            positions[k][2] = yStart;
            positions[k][3] = yEnd;
            
            // Notes!!!!
            // Need to make an array of values outside the for loop because they may be overwritten on the enxt iteration.

            ((void **)argsArr[k])[0] = &positions[k][0];     
            ((void **)argsArr[k])[1] = &positions[k][1];
            ((void **)argsArr[k])[2] = &positions[k][2];
            ((void **)argsArr[k])[3] = &positions[k][3];
            ((void **)argsArr[k])[4] = &xMin;
            ((void **)argsArr[k])[5] = &yMin;
            ((void **)argsArr[k])[6] = &thresh;
            ((void **)argsArr[k])[7] = &iter;
            ((void **)argsArr[k])[8] = &ySlope;
            ((void **)argsArr[k])[9] = &xSlope;
            ((void **)argsArr[k])[10] = matrix; 
        }
    }

    for(int i  = 0; i < xDivs; i++){
        for(int j = 0; j < yDivs; j++){
            int k = i*xDivs + j;
            pthread_create(&pthreadIDs[k], NULL, myThreadFun, (void *)argsArr[k]);
        }
    }

    for(int i  = 0; i < xDivs; i++){
        for(int j = 0; j < yDivs; j++){
            int k = i*xDivs + j;
            pthread_join(pthreadIDs[k], NULL);
            free(argsArr[k]);
            // printf("Join %d\n",k);
        }
    }
    for(uint i = 0; i < xDivs*yDivs; i++){
        free(positions[i]);
        //again,check for NULL
    }
    free(positions);
    free(argsArr);
    free(pthreadIDs);
    sem_destroy(&mutex);

    FILE* pgmimg;
    // pgmimg = fopen("pgmimg.pgm", "wb");
    size_t fileTime = time(NULL);
    size_t fileSize = snprintf(NULL, 0, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime) + 1;
    size_t fileSize2 = snprintf(NULL, 0, "%d_%d_%g_%lu.jpeg", xResolution, yResolution, thresh, fileTime) + 1;
    size_t fileSize3 = snprintf(NULL, 0, "%d_%d_%g_%lu.png", xResolution, yResolution, thresh, fileTime) + 1;
    char* filename = malloc(fileSize);
    char* filename2 = malloc(fileSize2);
    char* filename3 = malloc(fileSize3);
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
            fprintf(pgmimg, "%d ",matrix[i][j]);
            if(debug == 1){
                printf("%3d",matrix[i][j]);
            }
        }
        if(debug == 1){
            printf("\n");
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);


    char *data;
    makeColourfull(&data, matrix, xResolution, yResolution, iter);
    printf("Value: %d %d %d\n",data[0],data[xResolution*yResolution/2 + xResolution/2],data[xResolution*yResolution]);
    // stbi_write_jpg(filename2,xResolution,yResolution,1,data,100);
    stbi_write_jpg(filename2,xResolution,yResolution,3,data,90);
    // stbi_write_png(filename3,xResolution,yResolution,3,data,0);
    
    

    for(uint i=0;i<xResolution;i++){
        free(matrix[i]);
    }
    free(matrix);
    free(data);
    free(filename2);
    free(filename3);
}

void makeColourfull(char **target, uint **matrix, uint xResolution, uint yResolution, uint iter){
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
        blackredorangewhite[i + 256 + 128].r = 255;
        blackredorangewhite[i + 256 + 128].g = 128 + i;
        blackredorangewhite[i + 256 + 128].b = 2*i;
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
    printf("max = %d\n",maxValue);
    
    *target = malloc(xResolution*yResolution*sizeof(char)*3);
    char *data = *target;
    uint index = 0;
    uint index2 = 0;
    for(uint i = 0; i < xResolution; i++){
        for(uint j = 0; j < yResolution; j++){
            index = (int)(((long)matrix[i][j] * colorSelected.numEntries) / maxValue);
            index2 = 3*(i + j*xResolution);
            
            data[index2 + 0] = colorSelected.selection[index].r;
            data[index2 + 1] = colorSelected.selection[index].g;
            data[index2 + 2] = colorSelected.selection[index].b;
        }
    }
}

/*
    parameters:
        c: complex number representing the grid square currently being mapped.
        thresh: threshold for the set that determines how quickly coordinated are discarded.
        iterationMax: how many time to loop if coordinate is part of the set.
*/
uint doMath(double c_real, double c_imag , double thresh, uint iterationMax){
    // Zn_1 = Zn^2 + c;
    complex C, Zn, var1, var2;
    C.real = c_real;
    C.imag = c_imag;
    Zn.real = 0;
    Zn.imag = 0;
    var1.real = 2;
    var1.imag = 0.25;
    var2.real = 2;
    var2.imag = 2;
    uint numberOfIterations = 0;
    if(debug == 1){
        printf("Zn = %f + i%f\n",Zn.real,Zn.imag);
    }
    // printf("Thresh = %d\n",(uint)thresh);
    while(complexMag(&Zn) <= (uint)thresh && numberOfIterations < iterationMax){
        numberOfIterations++;
        complexAdd(&Zn,&C);
        complexPower(&Zn,2);
        complexMult(&Zn,&var2);
        // complexPower(&Zn,4);
        // complexPower(&Zn,6);
        // complexPower(&Zn,8);
        

    }
    numberOfIterations--;

    return numberOfIterations;
}

uint complexMag(complex *target){
    
    return sqrt(pow(target->real,2) + pow(target->imag,2));
}

void complexPower(complex *target, uint power){
    double r = sqrt(pow(target->real,2) + pow(target->imag,2));
    double theta = atan(target->imag/target->real);
    target->real = pow(r,power)*cos(power*theta);
    target->imag = pow(r,power)*sin(power*theta);
}

void complexAdd(complex *target, complex *addition){
    target->real += addition->real;
    target->imag += addition->imag;
}

void complexMult(complex *target, complex *mult){
    target->real = target->real * mult->real - target->imag * mult->imag;
    mult->imag = target->real * mult->imag + target->imag * mult->real;
}
