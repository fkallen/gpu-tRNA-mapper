#ifndef LOCAL_ALIGNMENT_API_CUH
#define LOCAL_ALIGNMENT_API_CUH

#include "local_alignment/kernels_scoreonly.cuh"
#include "util.cuh"
#include "cuda_errorcheck.cuh"
#include "substitutionmatrix.cuh"

#include <type_traits>
#include <map>

#include <thrust/device_vector.h>



template<
    int alphabetSize_,
    class ScoreType_, 
    PenaltyType penaltyType_,
    int blocksize_,
    int groupsize_, 
    int numItems_
>
class LocalAlignment_32bit{
public:
    using ScoreType = ScoreType_;
    static constexpr AlignmentType alignmentType = AlignmentType::LocalAlignment;
    static constexpr int alphabetSize = alphabetSize_;
    static constexpr PenaltyType penaltyType = penaltyType_;
    static constexpr int blocksize = blocksize_;
    static constexpr int groupsize = groupsize_;
    static constexpr int numItems = numItems_;
private:    
    static_assert(alphabetSize > 1);
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

    static constexpr int alphabetSizeWithPadding = alphabetSize+1;
    static constexpr int tileSize = groupsize * numItems;

public:

    using GpuSubstitutionMatrixType = GpuSubstitutionMatrix<AlignmentType::LocalAlignment, ScoreType, alphabetSizeWithPadding>;


public:
    LocalAlignment_32bit(){
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    }

    constexpr bool isShortQuery(int queryLength){
        return queryLength <= tileSize;
    }


    /*
        Returns the size of temp storage required to fit 1 thread block of size blocksize per SM.
        Using a temp storage of smaller size will underutilize the GPU.

        Some combinations of gpu architecture / groupsize / numItems / blocksize could fit multiple
        thread blocks per SM. In those cases it can improve performance if a multiple of the
        return number of bytes is used.
    */
    size_t getMinimumSuggestedTempBytes_longQuery(int maximumSubjectLength) const{
        constexpr int groupsPerBlock = (blocksize / groupsize);
        const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumSubjectLength);        
        const size_t tempBytes1BlockPerSM = tileTempBytesPerGroup * groupsPerBlock * numSMs;
        return tempBytes1BlockPerSM;
    }

    template<
        class SequenceInputDataOneToOne
    >
    void oneToOne_shortQuery(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& inputData, //each query must have length <= groupsize * numItems
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(!SequenceInputDataOneToOne::isSameQueryForAll);

        //setup and call alignment kernel
        using SUBMAT = typename GpuSubstitutionMatrixType::Submat32;
        auto kernel = localalignment::alphabet_substitutionmatrix_floatOrInt_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToOne,
            SUBMAT
        >;

        int smem = sizeof(SUBMAT);
        if(smem > 48*1024){
            cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if(status != cudaSuccess){
                cudaGetLastError(); //reset error state
                throw std::runtime_error("Could not set shared memory kernel attribute");
            }
        }
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));
        constexpr int groupsPerBlock = (blocksize / groupsize);
        constexpr int alignmentsPerBlock = groupsPerBlock;
        const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
        const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

        const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            gpuSubstitutionMatrix.getSubmat32(),
            scoring
        );
        CUDACHECKASYNC;
    }

    template<
        class SequenceInputDataOneToOne
    >
    void oneToOne_longQuery(
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& inputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(!SequenceInputDataOneToOne::isSameQueryForAll);

        if(d_temp == nullptr || tempBytes == 0) throw std::runtime_error("tempstorage is 0");

        const int maximumTargetLength = inputData.getMaximumSubjectLength();

        //setup and call alignment kernel
        using SUBMAT = typename GpuSubstitutionMatrixType::Submat32;
        auto kernel = localalignment::alphabet_substitutionmatrix_floatOrInt_multipass_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToOne,
            SUBMAT
        >;

        int smem = sizeof(SUBMAT);
        if(smem > 48*1024){
            cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if(status != cudaSuccess){
                cudaGetLastError(); //reset error state
                throw std::runtime_error("Could not set shared memory kernel attribute");
            }
        }
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));

        const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumTargetLength);

        constexpr int groupsPerBlock = (blocksize / groupsize);
        constexpr int alignmentsPerBlock = groupsPerBlock;
        const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
        const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;
        const int maxNumBlocksByTempBytes = tempBytes / (tileTempBytesPerGroup * groupsPerBlock);

        const int numBlocks = std::min(maxNumBlocksByTempBytes, std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy));
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            gpuSubstitutionMatrix.getSubmat32(),
            scoring,
            d_temp,
            tileTempBytesPerGroup
        );
        CUDACHECKASYNC;
    }

private:

    size_t getTileTempBytesPerGroup(int maximumTargetLength) const{
        using TempStorageDataType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            ScoreType,
            typename Vectorized2<ScoreType>::type
        >::type;
        const int maximumTargetLengthPadded = maximumTargetLength + groupsize;
        const size_t tileTempBytesPerGroup = sizeof(TempStorageDataType) * maximumTargetLengthPadded;
        return tileTempBytesPerGroup;
    }



private:
    int deviceId;
    int maxSharedMemoryPerBlockOptin;
    int numSMs;
};













#endif
