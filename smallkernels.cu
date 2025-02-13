#include "smallkernels.cuh"

#include "cuda_errorcheck.cuh"
#include "hpc_helpers/hpc_helpers.h"
#include <cstdio>

#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <rmm/exec_policy.hpp>
#include <rmm/device_uvector.hpp>


__global__
void verticalMaxReduceWithIndexKernel(const int* __restrict__ input, int numRows, int numColumns, int* __restrict__ maxOutput, int* __restrict__ rowIndexOfMax){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int c = tid; c < numColumns; c += stride){
        int myMax = input[0 * numColumns + c];
        // printf("(%d,%d) ", 0, input[0 * numColumns + c]);
        int maxIndex = 0;
        for(int r = 1; r < numRows; r++){
            const int next = input[r * numColumns + c];
            // printf("(%d,%d) ", r, input[r * numColumns + c]);
            if(next > myMax){
                myMax = next;
                maxIndex = r;
            }
        }
        // printf("\n");
        maxOutput[c] = myMax;
        rowIndexOfMax[c] = maxIndex;

        //check how many equal best there are
        // int equal = 0;
        // for(int r = 0; r < numRows; r++){
        //     const int next = input[r * numColumns + c];
        //     if(next == myMax){
        //         equal++;
        //     }
        // }
        // if(equal > 1){
        //     if(myMax >= 10){
        //         printf("equal %d, %d\n", equal, myMax);
        //     }
        // }
    }
}


void callVerticalMaxReduceWithIndexKernel(
    const int* d_input, 
    int numRows, 
    int numColumns, 
    int* d_maxOutput, 
    int* d_rowIndexOfMax,
    cudaStream_t stream
){
    verticalMaxReduceWithIndexKernel<<<SDIV(numColumns, 128), 128, 0, stream>>>(
        d_input,
        numRows,
        numColumns,
        d_maxOutput,
        d_rowIndexOfMax
    );
    CUDACHECKASYNC
}




__global__
void verticalMaxReduceFindNumBestScoresKernel(const int* __restrict__ input, int numRows, int numColumns, int* __restrict__ maxOutput, int* __restrict__ numMax, int minScore){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int c = tid; c < numColumns; c += stride){
        int myMax = input[0 * numColumns + c];
        // printf("(%d,%d) ", 0, input[0 * numColumns + c]);
        for(int r = 1; r < numRows; r++){
            const int next = input[r * numColumns + c];
            // printf("(%d,%d) ", r, input[r * numColumns + c]);
            if(next > myMax){
                myMax = next;
            }
        }
        // printf("\n");
        maxOutput[c] = myMax;

        //check how many equal best there are
        int equal = 0;
        if(minScore <= myMax){
            for(int r = 0; r < numRows; r++){
                const int next = input[r * numColumns + c];
                if(next == myMax){
                    equal++;
                }
            }
        }
        numMax[c] = equal;
    }
}

void callVerticalMaxReduceFindNumBestScoresKernel(
    const int* d_input, 
    int numRows, 
    int numColumns, 
    int* d_maxOutput, 
    int* d_numMax, 
    int minScore,
    cudaStream_t stream
){
    verticalMaxReduceFindNumBestScoresKernel<<<SDIV(numColumns, 128), 128, 0, stream>>>(
        d_input,
        numRows,
        numColumns,
        d_maxOutput,
        d_numMax,
        minScore
    );
    CUDACHECKASYNC
}


__global__
void verticalMaxReduceFindIndicesOfBestScoresKernel(
    const int* __restrict__ input, 
    int numRows, 
    int numColumns, 
    const int* __restrict__ maxOutput, 
    const int* __restrict__ numMax, 
    const int* __restrict__ numMaxPrefixSum,
    int minScore,
    int* __restrict__ rowIndicesOfMax
){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int c = tid; c < numColumns; c += stride){
        int myMax = maxOutput[c];
        if(minScore <= myMax){
            int* myIndexOutputBuffer = rowIndicesOfMax + numMaxPrefixSum[c];
            int numOutputs = 0;
            for(int r = 0; r < numRows; r++){
                const int next = input[r * numColumns + c];
                if(next == myMax){
                    myIndexOutputBuffer[numOutputs++] = r;
                }
            }
        }
    }
}

void callVerticalMaxReduceFindIndicesOfBestScoresKernel(
    const int* d_input, 
    int numRows, 
    int numColumns, 
    const int* d_maxOutput, 
    const int* d_numMax, 
    const int* d_numMaxPrefixSum,
    int minScore,
    int* d_rowIndicesOfMax,
    cudaStream_t stream
){
    verticalMaxReduceFindIndicesOfBestScoresKernel<<<SDIV(numColumns, 128), 128, 0, stream>>>(
        d_input,
        numRows,
        numColumns,
        d_maxOutput,
        d_numMax,
        d_numMaxPrefixSum,
        minScore,
        d_rowIndicesOfMax
    );
    CUDACHECKASYNC
}


// __global__
// void horizontalMaxReduceWithIndexKernel(const int* __restrict__ input, int numRows, int numColumns, int* __restrict__ maxOutput, int* __restrict__ rowIndexOfMax){

//     auto warp = cg::tiled_partition<32>(cg::this_thread_block());
//     const int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     const int warpId = tid / 32;
//     const int numWarpsInGrid = (blockDim.x * gridDim.x) / 32;

//     for(int r = warpId; r < numRows; r += numWarpsInGrid){
//         int myMax = -999999;
//         int maxIndex = -1;
//         for(int c = warp.thread_rank(); c < numColumns; c += warp.size()){
//             const int next = input[r * numColumns + c];
//             // printf("(%d, %d) ", c, input[r * numColumns + c]);
//             if(next > myMax){
//                 myMax = next;
//                 maxIndex = c;
//             }
//         }
//         // warp.sync();
//         // if(warp.thread_rank() == 0){
//         //     printf("\n");
//         // }
//         if(maxIndex >= 51) printf("error maxIndex\n");
//         int2 packed = make_int2(myMax, maxIndex);
//         int2 reduced = cg::reduce(warp, packed, []__device__(int2 l, int2 r){
//             if(l.x > r.x) return l;
//             else return r;
//         });

//         if(warp.thread_rank() == 0){
//             maxOutput[r] = reduced.x;
//             if(reduced.y >= 51) printf("error reduced.y = %d\n", reduced.y );
//             rowIndexOfMax[r] = reduced.y;
//         }
//     }
// }




void getSegmentIdsPerElement(
    int* d_output,
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream
){
    auto thrustpolicy = rmm::exec_policy_nosync(stream);

    CUDACHECK(cudaMemsetAsync(d_output, 0, sizeof(int) * numElements, stream));
    //must not scatter for empty segments
    thrust::scatter_if(
        thrustpolicy,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + numSegments, 
        d_segmentSizesPrefixSum,
        thrust::make_transform_iterator(
            d_segmentSizes, 
            [] __host__ __device__ (int i){return i > 0;}
        ),
        d_output
    );

    thrust::inclusive_scan(
        thrustpolicy,
        d_output, 
        d_output + numElements, 
        d_output, 
        thrust::maximum<int>{}
    );
}

rmm::device_uvector<int> getSegmentIdsPerElement(
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream
){
    rmm::device_uvector<int> result(numElements, stream);

    getSegmentIdsPerElement(
        result.data(),
        d_segmentSizes,
        d_segmentSizesPrefixSum,
        numSegments,
        numElements,
        stream
    );

    return result;
}