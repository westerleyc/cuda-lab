#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */


void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = 1.0f;
    }

    return;
}

__global__ void kernel1(float *A, float *B, unsigned int *C, const int N)
{
    float x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c1,c2;
     x = idx;
   
    
    asm("mov.u32 %0,%%clock;" : "=r"(c1));
    asm("div.rn.f32 %0,%1,%2;": "=f"(x): "f"(x),"f"(x));
    asm("mov.u32 %0,%%clock;" : "=r"(c2));
    if ( idx < N) { B[idx] = x+A[idx]; C[idx] =  c2-c1; }
}

__global__ void kernel(float *A, float *B, unsigned int *C, const int N)
{
    float x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c1,c2;
     x = A[idx];
    // x = x/x;
    
    asm("mov.u32 %0,%%clock;" : "=r"(c1));
    asm("add.f32 %0,%1,%2;": "=f"(x): "f"(x),"f"(x));
    asm("mov.u32 %0,%%clock;" : "=r"(c2));
    if ( idx < N) { B[idx] = x; C[idx] =  c2-c1; }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 26;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef;
    unsigned int *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (unsigned int *)malloc(nBytes);

  
    // initialize data at host side

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    
    // malloc device global memory
    float *d_A, *d_B;
    unsigned int *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((unsigned int**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
  
    // invoke kernel at host side
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

 
    kernel<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
  
    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    float average = 0;
    for (int j=0; j< nElem/1000; j++) {
      float local = 0;
      for (int i=0; i < 1000; i++) {
       if ( (1000*j+i)%100000 == 0) printf(" C= %d\n",gpuRef[1000*j+i]);
       local += gpuRef[1000*j+i];
      }
      average += local;
    }
average = average/(1.0*nElem);
    printf(" vetor %d average cycles= %f\n",nElem,average);
    
    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
