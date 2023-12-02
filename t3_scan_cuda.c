
//NVDIA GPU Gems 3: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//Cuda Toolkit Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction

/* - Assignment comments -

    Modifications to solve the Shared Memory Bank Conflict (SMBC)
The memory access pattern causes SMBC to happen.
The shared memory treated in this type of algorithm is made up of several "banks".
If several threads in the same warp access the same bank, there is a conflict.
Conflicts of this type generate serialization of the multiple accesses to the memory bank.
This means that a shared memory access with a memory bank conflict of degree 'n',
needs 'n' times more cycles to process, compared to an access without conflict.

Memory bank conflicts are avoidable here by being careful with memory access of vectors of type __shared__
The index value divided by the number of shared memory banks is added to the index.
*/
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \     ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void prescan(float *g_odata, float *g_idata, int n) { 

    extern __shared__ float temp[];  // Author comment: allocated on invocation 
    int thid = threadIdx.x; 
    int offset = 1;
    //Assignment comment: Bloco A 
    //Author comment: temp[2*thid] = g_idata[2*thid]; // load input into shared memory 
    //Author comment: temp[2*thid+1] = g_idata[2*thid+1]; 
    //Assignment comment: ---- Solve SMBC -----
    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];
    //---------- 

    for (int d = n>>1; d > 0; d >>= 1)   // Author comment: build sum in place up the tree 
    { 
        __syncthreads();

        if (thid < d)    
        { 
            //Assignment comment: Bloco B 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;
            ai += ai / NUM_BANKS; 
            bi += bi / NUM_BANKS; 
            temp[bi] += temp[ai];    
        }
        offset *= 2; 
    } 

    //Assignment comment: Bloco C 
    if (thid == 0) 
    { 
        //Author comment: temp[n - 1] = 0; 
        //Assignment comment: ---- Solve SMBC ----
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        //-----------

    } // Author comment: clear the last element 

    for (int d = 1; d < n; d *= 2) // Author comment: traverse down tree & build scan 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)      
        {
            //Assignment comment: Bloco D 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        } 
    }

    __syncthreads();

    //Assignment comment: Bloco E 
    //Author comment: g_odata[2*thid] = temp[2*thid]; // write results to device memory
    //Author comment: g_odata[2*thid+1] = temp[2*thid+1];

    //Assignment comment: ---- Solve SMBC ----
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
    //---------
}

