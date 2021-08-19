// #include <complex.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct complex {
    double real;
    double imag;
} complex;

void doSomething(int, int, double, double, double, double ,double, int);
int doMath(double, double,double, int);
int complexMag(complex*);
void complexPower(complex*, int);
void complexAdd(complex*, complex*);

int debug = 0;

int main(int argc, char *argv[]){

    // !!! Toggle print statement debugging on(1)/off(0). Default is off
    debug = 0;
    int yRes = 889;
    int xRes = 500;
    double thresh = 1;
    int iter = 127;
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

void doSomething(int xResolution, int yResolution, double xMin, double xMax, double yMin, double yMax, double thresh, int iter){
    // Add 1 to the xRes and yRes when creating a matrix to include the endpoint(s). Also, this affects how the code below is made, so I created a variable
    // instead of hardcoding the value throughout.
    int **matrix;
    int resOffset = 1;
    matrix=malloc((xResolution + resOffset) * sizeof(int*));
    // Check for NULL
    for(int i = 0; i < xResolution + resOffset; i++){
        matrix[i]=malloc((yResolution + resOffset) * sizeof(int));
        //again,check for NULL
    }

    
    double xSlope = (double)(xMax - xMin) / xResolution;
    double ySlope = (double)(yMax - yMin) / yResolution;
    if(debug == 1){
        printf("Slopes: %f %f\n",xSlope, ySlope);
    }
    

    double tempX, tempY;
    for(int i = 0; i < (xResolution + resOffset); i++){
        tempX = xSlope*((double)i - 0) + xMin;
        for(int j = 0; j < (yResolution + resOffset); j++){
            tempY = ySlope*(j - 0) + yMin;
            if(debug == 1){
                printf("%d %d : %f %f\n",i,j,tempX,tempY);
            }
            matrix[i][j] = doMath(tempX, tempY, thresh, iter);
        }
    }

    FILE* pgmimg;
    // pgmimg = fopen("pgmimg.pgm", "wb");
    size_t fileTime = time(NULL);
    size_t fileSize = snprintf(NULL, 0, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime) + 1;
    char* filename = malloc(fileSize);
    snprintf(filename, fileSize, "%d_%d_%g_%lu.pgm", xResolution, yResolution, thresh, fileTime);
    pgmimg = fopen(filename, "a");
    free(filename);
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n",xResolution + resOffset,yResolution + resOffset);
    fprintf(pgmimg, "%d\n",iter);
    
    for(int j = 0; j < yResolution + resOffset; j++){
        for(int i = 0; i < xResolution + resOffset; i++){
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

    for(int i=0;i<xResolution;i++){
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
int doMath(double c_real, double c_imag , double thresh, int iterationMax){
    // Zn_1 = Zn^2 + c;
    complex C, Zn;
    C.real = c_real;
    C.imag = c_imag;
    Zn.real = 0;
    Zn.imag = 0;
    int numberOfIterations = -1;
    if(debug == 1){
        printf("Zn = %f + i%f\n",Zn.real,Zn.imag);
    }
    while(complexMag(&Zn) <= thresh && numberOfIterations < iterationMax){
        numberOfIterations++;
        complexAdd(&Zn,&C);
        complexPower(&Zn,2);
        // complexPower(&Zn,4);

    }

    return numberOfIterations;
}

int complexMag(complex *target){
    
    return sqrt(pow(target->real,2) + pow(target->imag,2));
}

void complexPower(complex *target, int power){
    double r = sqrt(pow(target->real,2) + pow(target->imag,2));
    double theta = atan(target->imag/target->real);
    target->real = pow(r,power)*cos(power*theta);
    target->imag = pow(r,power)*sin(power*theta);
}

void complexAdd(complex *target, complex *addition){
    target->real += addition->real;
    target->imag += addition->imag;
}