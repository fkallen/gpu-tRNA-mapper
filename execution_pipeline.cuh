#ifndef EXECUTION_PIPELINE_CUH
#define EXECUTION_PIPELINE_CUH

#include "raii_cudaevent.cuh"

#include <vector>
#include <string>
#include <future>

#include "common.cuh"
#include "parasail_helpers.hpp"

struct BatchData{
    size_t batchId = 0;
    const SequenceCollection* referenceSequences;
    SequenceCollection readSequences;

    CudaEvent h2d_Start;
    CudaEvent h2d_Stop;
    CudaEvent otherkernels_Start;
    CudaEvent otherkernels_Stop;
    CudaEvent alignmentkernels_Start;
    CudaEvent alignmentkernels_Stop;
    CudaEvent d2h_Start;
    CudaEvent d2h_Stop;

    size_t computedCells{};
    std::vector<int, PinnedAllocator<int>> h_permutationlist{};
    std::vector<int, PinnedAllocator<int>> h_scores{};
    std::vector<int, PinnedAllocator<int>> h_readIndexPerReferenceSequence{};
    std::vector<int, PinnedAllocator<int>> h_referenceSequenceIdsOfMax{};
    std::vector<int, PinnedAllocator<int>> h_numReferenceSequenceIdsPerRead{};
    std::vector<int, PinnedAllocator<int>> h_numReferenceSequenceIdsPerReadPrefixSum{};
    std::vector<int, PinnedAllocator<int>> h_queryStartPositions_inclusive{};
    std::vector<int, PinnedAllocator<int>> h_subjectStartPositions_inclusive{};
    std::vector<int, PinnedAllocator<int>> h_queryEndPositions_inclusive{};
    std::vector<int, PinnedAllocator<int>> h_subjectEndPositions_inclusive{};
    std::vector<ParasailResultData> parasailResultData{};
    std::vector<std::string> samresultStrings{};

    void clear(){
        readSequences.clear();
        h_permutationlist.clear();
        h_scores.clear();
        h_readIndexPerReferenceSequence.clear();
        h_referenceSequenceIdsOfMax.clear();
        h_numReferenceSequenceIdsPerRead.clear();
        h_numReferenceSequenceIdsPerReadPrefixSum.clear();
        h_queryStartPositions_inclusive.clear();
        h_subjectStartPositions_inclusive.clear();
        h_queryEndPositions_inclusive.clear();
        h_subjectEndPositions_inclusive.clear();
        parasailResultData.clear();
        computedCells = 0;
    }

};

using BatchDataQueue = SimpleConcurrentQueue<BatchData*>;



std::future<void> launchReadParser(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    int deviceId
);

std::future<void> launchOutputWriter(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    int deviceId
);

std::future<void> launchCPUTracebackWorker(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    const std::vector<std::vector<int>>* substitutionMatrix2D,
    parasail_matrix_t* parasailScoringMatrix
);

std::future<void> launchLocalAlignmentGPUTopscoresWorker(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    const std::vector<std::vector<int>>* substitutionMatrix2D,
    parasail_matrix_t* parasailScoringMatrix,
    int deviceId
);

std::future<void> launchSemiglobalAlignmentGPUTopscoresWorker(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    const std::vector<std::vector<int>>* substitutionMatrix2D,
    parasail_matrix_t* parasailScoringMatrix,
    int deviceId
);



#endif