
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
using namespace std;

#define N 1024
#define THREADS_PER_BLOCK 256
//#define DOTS_NUM 1024 //chislo tochek v kvadrate



//time(null);
__global__ void eval(int *Ncirc, float * x, float *y)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < N)
	{
		if (x[i] * x[i] + y[i] * y[i] <= 1)
		{
			atomicAdd(Ncirc, 1);
		}
	}
}

__global__ void evalx(float * x, unsigned seed)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	curandState_t t;
	curand_init(seed, i, 0 , &t);
	if (i < N)
	{
		x[i] = curand_uniform(&t);
		//printf("x[%i]=%f\n", i, x[i]);
	}
}
__global__ void evaly(float * y, unsigned seed)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	curandState_t m;
	curand_init(seed, i, 0, &m);
	if (i < N)
	{
		y[i] = curand_uniform(&m);
		//printf("y[%i]=%f\n", i, y[i]);
	}
}

int main()
{

	int sizeI = sizeof(int);
	int sizeF = sizeof(float);
	
	int Ncirc = 0;
	float x[N];
	float y[N];

	int *d_Ncirc;
	float *d_x, *d_y;

	cudaMalloc((void**)&d_x, N*sizeF);
	cudaMalloc((void**)&d_y, N*sizeF);
	cudaMalloc((void**)&d_Ncirc, sizeI);
	cudaMemcpy(d_Ncirc, &Ncirc, sizeI, cudaMemcpyHostToDevice);

	evalx << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>> > (d_x, 0);
	evaly << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_y, time(NULL));
	cudaMemcpy(&x, d_x, N*sizeF, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y, d_y, N*sizeF, cudaMemcpyDeviceToHost);
	for (int i = 0; i != N; i++)
	{
		printf("x[%i]=%f\n", i, x[i]);
		printf("y[%i]=%f\n", i, y[i]);
	}
	eval << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_Ncirc,d_x,d_y);
	cudaMemcpy(&Ncirc, d_Ncirc, sizeI, cudaMemcpyDeviceToHost);
	printf("pi = %f\n", (Ncirc*4.0)/N);
	cudaFree(d_Ncirc);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaDeviceSynchronize();
	getchar();
}
