#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "stb_image_write.h"

typedef struct complex {
    double real;
    double imag;
} complex;

void doSomething(uint, uint, double, double, double, double ,double, uint);
uint doMath(double, double,double, uint);
uint complexMag(complex*);
void complexPower(complex*, uint);
void complexAdd(complex*, complex*);

uint debug = 0;

int main(int argc, char *argv[]){
    // ./run.bin 1920 1080 8 255 1
    // !!! Toggle print statement debugging on(1)/off(0). Default is off
    debug = 0;
    uint yRes = 500;
    uint xRes = 889;
    double thresh = 1;
    uint iter = 127;
    double xMin = -2;
    double xMax = 1;
    double yMin = -1.5;
    double yMax = 1.5;
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

    clock_t start = clock(), diff;
    doSomething(xRes, yRes, xMin, xMax, yMin, yMax, thresh, iter);
    diff = clock() - start;

    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    return 1;

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
    

    double tempX, tempY;
    for(uint i = 0; i < (xResolution + resOffset); i++){
        tempX = xSlope*((double)i - 0) + xMin;
        for(uint j = 0; j < (yResolution + resOffset); j++){
            tempY = ySlope*(j - 0) + yMin;
            if(debug == 1){
                printf("%d %d : %f %f\n",i,j,tempX,tempY);
            }
            matrix[i][j] = doMath(tempX, tempY, thresh, iter);
        }
    }

    FILE* pgmimg;
    FILE* jpgimg;
    // pgmimg = fopen("pgmimg.pgm", "wb");
    size_t fileTime = time(NULL);
    size_t fileSize = snprintf(NULL, 0, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime) + 1;
    size_t fileSize2 = snprintf(NULL, 0, "%d_%d_%g_%lu.jpg", xResolution, yResolution, thresh, fileTime) + 1;
    char* filename = malloc(fileSize);
    char* filename2 = malloc(fileSize2);
    snprintf(filename, fileSize, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime);
    snprintf(filename2, fileSize2, "%d_%d_%g_%lu.jpg", xResolution, yResolution, thresh, fileTime);
    
    pgmimg = fopen(filename, "a");
    jpgimg = fopen(filename2, "a");
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

    // Vertical
    // char *data = malloc(xResolution*yResolution*sizeof(char));
    // for(uint j = 0; j < yResolution; j++){
    //     for(uint i = 0; i < xResolution; i++){
    //         data[i*yResolution + j] = (char)matrix[i][j];
    //     }
    // }
    // printf("Value: %d %d %d\n",data[0],data[xResolution*yResolution/2],data[xResolution*yResolution]);
    // stbi_write_jpg(filename2,yResolution,xResolution,1,data,100);

    char *data = malloc(xResolution*yResolution*sizeof(char)*3);
    // char *data = malloc(xResolution*yResolution*sizeof(char));
    for(uint i = 0; i < xResolution; i++){
        for(uint j = 0; j < yResolution; j++){
            // data[i + j*xResolution] = (char)matrix[i][j];
            data[3*(i + j*xResolution) + 0] = (char)matrix[i][j];
            data[3*(i + j*xResolution) + 1] = 0;
            data[3*(i + j*xResolution) + 2] = (char)matrix[i][j];
        }
    }
    printf("Value: %d %d %d\n",data[0],data[xResolution*yResolution/2],data[xResolution*yResolution]);
    // stbi_write_jpg(filename2,xResolution,yResolution,1,data,100);
    stbi_write_jpg(filename2,xResolution,yResolution,3,data,100);

    
    fclose(jpgimg);

    for(uint i=0;i<xResolution;i++){
        free(matrix[i]);
    }
    free(matrix);
}


/*
    parameters:
        c: complex number representing the grid square currently being mapped.
        thresh: threshold for the set that determines how quickly coordinated are discarded.
        iterationMax: how many time to loop if coordinate is part of the set.
*/
uint doMath(double c_real, double c_imag , double thresh, uint iterationMax){
    // Zn_1 = Zn^2 + c;
    complex C, Zn;
    C.real = c_real;
    C.imag = c_imag;
    Zn.real = 0;
    Zn.imag = 0;
    uint numberOfIterations = 0;
    if(debug == 1){
        printf("Zn = %f + i%f\n",Zn.real,Zn.imag);
    }
    while(complexMag(&Zn) <= thresh && numberOfIterations < iterationMax){
        numberOfIterations++;
        complexAdd(&Zn,&C);
        complexPower(&Zn,2);
        // complexPower(&Zn,4);

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