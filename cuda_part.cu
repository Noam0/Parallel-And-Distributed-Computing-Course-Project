#include <stdio.h>
#include <string.h>

#define BLOCK_DIM 1024 // number of threads in a block
#define DEBUG

typedef struct
{
    int score;
    int offset;
    int k;
    int seq_len;
    char seq[2000];
} Seq_Info;




__global__ void work_cuda(Seq_Info* device_array, int seq2_len,char* seq1,int seq1_len,int* score_table,int table_size);
int computeOnGPU(Seq_Info *seq_array, int arr_size,char* seq1, int seq1_length);

/* Here we do an inclusive scan of 'array' in place.
   'size' is the number of elements in 'array'.
   it should be a power of 2.
 
   We assume that 'array' is in shared memory so that there is no need to 
   copy it to shared memory here.
    */
__device__ void scan_plus(int *array, int size)
{
    for (unsigned int stride=1; stride <= size/2; stride *= 2) {
        int v;
        if (threadIdx.x >= stride) {
            v = array[threadIdx.x - stride];
        }
        __syncthreads(); /* wait until all threads finish reading 
		                    an element */

        if (threadIdx.x >= stride)
            array[threadIdx.x] += v;

        __syncthreads(); /* wait until all threads finish updating an
		                    element */

    }

} // scan_plus


__global__ void work_cuda(Seq_Info* device_array, int seq2_len,char* seq1,int seq1_len,int* score_table,int table_size)
{

    int tid = threadIdx.x+blockDim.x; // 0-1023

    __shared__ int scores[blockDim.x*BLOCK_DIM];


#ifdef DEBUG
    if(tid == 0)
        printf("%d --> IN WORK CUDA \n", tid);
#endif
    
    int calculations =  

    if(tid < calculations )
    {
        // work


    }else{
        score[tid]=0;
    }



/*
    int shortest = (n1>n2) ? n2 : n1;
    
    if (tid < shortest)
        flags[tid] = s1[tid] - s2[tid]; // 65 - 90 = negative
    else
        flags[tid] = 0;

    __syncthreads();  // wait until all threads write to flags

    scan_plus(flags, BLOCK_DIM);
    // now 'flags' holds the result of (inclusive)  scanning the original 'flags'

    __syncthreads(); // wait until all threads complete write to flags

    if (tid == 0)
        *result = 0; // initialize result
    
    if((tid == 0) && (flags[tid] != 0))
        *result = flags[tid];

    if((tid != 0) && (flags[tid-1] == 0) && (flags[tid] != 0) && (tid < shortest))
        *result = flags[tid]; 
*/
     

}


int computeOnGPU(Seq_Info *seq_array, int arr_size,char* seq1,int seq1_length, int* score_table,int table_size)
{
    // Error code to check return values for CUDA calls
    printf("HELLO CUDA\n");
    cudaError_t err = cudaSuccess;
    size_t size = arr_size * sizeof(seq_array[0]);




    // Allocate memory on GPU to copy the data from the host
    Seq_Info *device_array;
    err = cudaMalloc((void **)&device_array, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return 1;
    }
    // Copy data from host to the GPU memory
    err = cudaMemcpy(device_array, seq_array, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        return(1);
    }
    
    
    // Allocate memory on GPU to copy the TABLE from the host
    int *device_table;
    err = cudaMalloc((void **)&device_table, table_size*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy TABLE from host to the GPU memory
    err = cudaMemcpy(device_table, score_table, table_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        return(1);
    }






    // Allocate memory on GPU to copy the seq1 from the host
    int *device_seq1;
    err = cudaMalloc((void **)&device_seq1, seq1_length*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy seq1 from host to the GPU memory
    err = cudaMemcpy(device_seq1, seq1, seq1_length, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        return(1);
    }

    // Allocate memory on GPU to copy the seq1 from the host
    int *device_seq1;
    err = cudaMalloc((void **)&device_seq1, seq1_length*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy seq1 from host to the GPU memory
    err = cudaMemcpy(device_seq1, seq1, seq1_length, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        return(1);
    }






    dim3 blockDim;
    blockDim.x = BLOCK_DIM;
    

    for(int i=0 ; i < arr_size ; i++)
    {
        int blocks = (((seq_array[i].seq_len - seq1_length) * seq_array[i].seq_len) / blockDim.x) + 1 ; 
        work_cuda<<<blocks, blockDim.x>>>(device_array[i], device_array[i].seq_len, device_seq1, seq1_length,device_table, table_size);
        /* note: next lines may be executed before the kernel is done */
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch incrementByOne kernel -  %s\n", cudaGetErrorString(err));
            return(1);
        }

    }





    //int blocksPerGrid =(arr_size + threadsPerBlock - 1) / threadsPerBlock;
    





    // Copy the  result from GPU to the host memory.
    err = cudaMemcpy(seq_array, device_array, arr_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        return(1);
    }

    // Free allocated memory on GPU
    if (cudaFree(device_array) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        return(1);
    }
    if (cudaFree(device_table) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        return(1);
    }

    return 0;



}