#ifndef SMALLKERNELS_CUH
#define SMALLKERNELS_CUH

#include "smallkernels.cuh"

#include <rmm/device_uvector.hpp>


void callVerticalMaxReduceWithIndexKernel(
    const int* d_input, 
    int numRows, 
    int numColumns, 
    int* d_maxOutput, 
    int* d_rowIndexOfMax,
    cudaStream_t stream
);

void callVerticalMaxReduceFindNumBestScoresKernel(
    const int* d_input, 
    int numRows, 
    int numColumns, 
    int* d_maxOutput, 
    int* d_numMax, 
    int minScore,
    cudaStream_t stream
);

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
);


void getSegmentIdsPerElement(
    int* d_output,
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream
);

rmm::device_uvector<int> getSegmentIdsPerElement(
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream
);




#endif