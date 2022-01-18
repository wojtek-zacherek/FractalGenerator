#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
      printf("%d  %2.2f  %2.2f \n",i,x[i],y[i]);
    //   y[i] = a*x[i] + y[i];
    y[i] = y[2*i] + y[2*i + 1];
  }
}

int main(void)
{
  int N = 1;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(2*N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, 2*N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[2*i] = 2.0f;
    y[2*i+1] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, 2*N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, 2*N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(y[i]-4.0f));
    printf("i=%d y=%f\n",i,y[i]);
  }
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}