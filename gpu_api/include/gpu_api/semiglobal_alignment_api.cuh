#ifndef SEMIGLOBAL_ALIGNMENT_API_CUH
#define SEMIGLOBAL_ALIGNMENT_API_CUH

#include "semiglobal_alignment/kernels_scoreonly.cuh"
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
class SemiglobalAlignment_32bit{
public:
    using ScoreType = ScoreType_;
    static constexpr AlignmentType alignmentType = AlignmentType::SemiglobalAlignment;
    static constexpr int alphabetSize = alphabetSize_;
    static constexpr PenaltyType penaltyType = penaltyType_;
    static constexpr int blocksize = blocksize_;
    static constexpr int groupsize = groupsize_;
    static constexpr int numItems = numItems_;
private:    
    static_assert(alphabetSize > 1);
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

    static constexpr int alphabetSizeWithPadding = alphabetSize;
    static constexpr int tileSize = groupsize * numItems;

public:

    using GpuSubstitutionMatrixType = GpuSubstitutionMatrix<AlignmentType::SemiglobalAlignment, ScoreType, alphabetSizeWithPadding>;


public:
    SemiglobalAlignment_32bit(){
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
        class SequenceInputDataOneToAll
    >
    void oneToAll_shortQuery(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData, //each query must have length <= groupsize * numItems
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(SequenceInputDataOneToAll::isSameQueryForAll);

        //setup and call alignment kernel
        using SUBMAT = typename GpuSubstitutionMatrixType::Submat32;
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_floatOrInt_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToAll,
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
        class SequenceInputDataOneToAll
    >
    void oneToAll_longQuery(
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(SequenceInputDataOneToAll::isSameQueryForAll);

        if(d_temp == nullptr || tempBytes == 0) throw std::runtime_error("tempstorage is 0");

        const int maximumTargetLength = inputData.getMaximumSubjectLength();

        //setup and call alignment kernel
        using SUBMAT = typename GpuSubstitutionMatrixType::Submat32;
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_floatOrInt_multipass_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToAll,
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
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_floatOrInt_kernel<
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
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_floatOrInt_multipass_kernel<
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







template<
    int alphabetSize_,
    class ScoreType_, 
    PenaltyType penaltyType_,
    int blocksize_,
    int groupsize_, 
    int numItems_
>
class SemiglobalAlignment_16x2bit{
public:
    using ScoreType = ScoreType_;
    static constexpr AlignmentType alignmentType = AlignmentType::SemiglobalAlignment;
    static constexpr int alphabetSize = alphabetSize_;
    static constexpr PenaltyType penaltyType = penaltyType_;
    static constexpr int blocksize = blocksize_;
    static constexpr int groupsize = groupsize_;
    static constexpr int numItems = numItems_;
private:
    static_assert(alphabetSize > 1);
    static_assert(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>);

    static constexpr int alphabetSizeWithPadding = alphabetSize;

public:
    using GpuSubstitutionMatrixType = GpuSubstitutionMatrix<AlignmentType::SemiglobalAlignment, typename ToScoreType32<ScoreType>::type, alphabetSizeWithPadding>;
private:

    static constexpr int tileSize = groupsize * numItems;

    using ScoreType16bit = typename ScalarScoreType<ScoreType>::type;

public: //strange compiler error if this is private
    struct KernelSubstitutionMatrixInformation{
        using SUBMAT_squared_oneToAll = typename GpuSubstitutionMatrixType::Submat16x2_SubjectSquaredQueryLinear;
        using SUBMAT_squared_oneToOne = typename GpuSubstitutionMatrixType::Submat16x2_SubjectSquaredQuerySquared;
        static constexpr SubstitutionMatrixDimensionMode dimMode_squared_oneToAll = SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear;
        static constexpr SubstitutionMatrixDimensionMode dimMode_squared_oneToOne = SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared;

        using SUBMAT_unpacked_oneToAll = typename GpuSubstitutionMatrixType::Submat16x2_SubjectLinearQueryLinear;
        using SUBMAT_unpacked_oneToOne = typename GpuSubstitutionMatrixType::Submat16x2_SubjectLinearQueryLinear;
        static constexpr SubstitutionMatrixDimensionMode dimMode_unpacked_oneToAll = SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear;
        static constexpr SubstitutionMatrixDimensionMode dimMode_unpacked_oneToOne = SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear;
    };
private:

    enum class SharedMemorySubstitutionMatrixApproach{
        NotSelected,
        Impossible,
        Squared,
        Unpacked
    };

public:



public:
    SemiglobalAlignment_16x2bit(){
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
        class SequenceInputDataOneToAll
    >
    void oneToAll_shortQuery(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData, //each query must have length <= groupsize * numItems
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(SequenceInputDataOneToAll::isSameQueryForAll);

        SharedMemorySubstitutionMatrixApproach sharedMemoryApproach 
            = selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_shortQuery<SequenceInputDataOneToAll>();

        if(sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Impossible){
            throw std::runtime_error("Could not set shared memory kernel attribute");
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Squared){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToAll;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_squared_oneToAll,
                SequenceInputDataOneToAll,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
            const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
            const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

            const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
            if(numBlocks <= 0){
                throw std::runtime_error("could not launch kernel. numBlocks <= 0");
            }

            kernel<<<numBlocks, blocksize, smem, stream>>>(
                d_scoreOutput,
                inputData,
                gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQueryLinear(),
                scoring
            );
            CUDACHECKASYNC;
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Unpacked){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToAll;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToAll,
                SequenceInputDataOneToAll,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
            const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
            const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

            const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
            if(numBlocks <= 0){
                throw std::runtime_error("could not launch kernel. numBlocks <= 0");
            }

            kernel<<<numBlocks, blocksize, smem, stream>>>(
                d_scoreOutput,
                inputData,
                gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
                scoring
            );
            CUDACHECKASYNC;
        }
    }

    #if 0
    template<
        class SequenceInputDataOneToAll
    >
    void oneToAll_shortQuery_auto(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        int queryLength,
        const SequenceInputDataOneToAll& inputData, //each query must have length <= groupsize * numItems
        Scoring1 scoring,
        cudaStream_t stream
    ){
        #if 1
            std::cout << "semiglobal alignment 16x2 oneToAll_shortQuery_auto not implemented\n";
        #else
        std::cout << "auto\n";
        static_assert(SequenceInputDataOneToAll::isSameQueryForAll);

        if(queryLength > tileSize) throw std::runtime_error("query too large for tileSize");

        using ScalarScoreType = typename ScalarScoreType<ScoreType>::type;
        using SmemSubmatOneToAllSquared = SharedSubstitutionMatrix<ScoreType, (alphabetSize+1)*(alphabetSize+1), (alphabetSize+1)>;
        using SmemSubmatOneToAllLinear = SharedSubstitutionMatrix<ScalarScoreType, (alphabetSize+1), (alphabetSize+1)>;

        using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToAll;
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_automatic_smem_layout_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToAll,
            SUBMAT
        >;

        int smem = sizeof(SmemSubmatOneToAllSquared);
        cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(status != cudaSuccess){
            smem = sizeof(SmemSubmatOneToAllLinear);
            status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if(status != cudaSuccess){
                cudaGetLastError(); //reset error state
                throw std::runtime_error("Could not set shared memory kernel attribute");
            }else{
                std::cout << "SmemSubmatOneToAllLinear\n";
            }
        }else{
            std::cout << "SmemSubmatOneToAllSquared\n";
        }


        int maxBlocksPerSM = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));
        constexpr int groupsPerBlock = (blocksize / groupsize);
        constexpr int alignmentsPerBlock = 2*groupsPerBlock;
        const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
        const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

        const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
            scoring
        );
        CUDACHECKASYNC;
        #endif
    }
    #endif

    template<
        class SequenceInputDataOneToAll
    >
    void oneToAll_longQuery(
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        static_assert(SequenceInputDataOneToAll::isSameQueryForAll);
        
        if(d_temp == nullptr || tempBytes == 0) throw std::runtime_error("tempstorage is 0");
        const int maximumTargetLength = inputData.getMaximumSubjectLength();

        SharedMemorySubstitutionMatrixApproach sharedMemoryApproach 
            = selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_longQuery<SequenceInputDataOneToAll>();

        if(sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Impossible){
            throw std::runtime_error("Could not set shared memory kernel attribute");
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Squared){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToAll;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_squared_oneToAll,
                SequenceInputDataOneToAll,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumTargetLength);

            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
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
                gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQueryLinear(),
                scoring,
                d_temp,
                tileTempBytesPerGroup
            );
            CUDACHECKASYNC;
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Unpacked){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToAll;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToAll,
                SequenceInputDataOneToAll,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumTargetLength);

            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
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
                gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
                scoring,
                d_temp,
                tileTempBytesPerGroup
            );
            CUDACHECKASYNC;
        }
    }

    #if 0
    template<
        class SequenceInputDataOneToOne
    >
    void oneToOne_shortQuery_auto(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& inputData, //each query must have length <= groupsize * numItems
        Scoring1 scoring,
        cudaStream_t stream
    ){
        #if 1
            std::cout << "semiglobal alignment 16x2 oneToOne_shortQuery_auto not implemented\n";
        #else
        std::cout << "auto\n";
        static_assert(!SequenceInputDataOneToOne::isSameQueryForAll);

        using ScalarScoreType = typename ScalarScoreType<ScoreType>::type;
        using SmemSubmatOneToOneSquared = SharedSubstitutionMatrix<ScoreType, (alphabetSize+1)*(alphabetSize+1), (alphabetSize+1)*(alphabetSize+1)>;
        using SmemSubmatOneToOneLinear = SharedSubstitutionMatrix<ScalarScoreType, (alphabetSize+1), (alphabetSize+1)>;

        using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToOne;
        auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_automatic_smem_layout_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            SequenceInputDataOneToOne,
            SUBMAT
        >;

        int smem = sizeof(SmemSubmatOneToOneSquared);
        cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(status != cudaSuccess){
            smem = sizeof(SmemSubmatOneToOneLinear);
            status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if(status != cudaSuccess){
                cudaGetLastError(); //reset error state
                throw std::runtime_error("Could not set shared memory kernel attribute");
            }else{
                std::cout << "SmemSubmatOneToOneLinear\n";
            }
        }else{
            std::cout << "SmemSubmatOneToOneSquared\n";
        }


        int maxBlocksPerSM = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));
        constexpr int groupsPerBlock = (blocksize / groupsize);
        constexpr int alignmentsPerBlock = 2*groupsPerBlock;
        const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
        const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

        const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
            scoring
        );
        CUDACHECKASYNC;
        #endif
    }
    #endif

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

        SharedMemorySubstitutionMatrixApproach sharedMemoryApproach 
            = selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_shortQuery<SequenceInputDataOneToOne>();

        if(sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Impossible){
            throw std::runtime_error("Could not set shared memory kernel attribute");
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Squared){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToOne;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_squared_oneToOne,
                SequenceInputDataOneToOne,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
            const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
            const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

            const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
            if(numBlocks <= 0){
                throw std::runtime_error("could not launch kernel. numBlocks <= 0");
            }

            kernel<<<numBlocks, blocksize, smem, stream>>>(
                d_scoreOutput,
                inputData,
                gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQuerySquared(),
                scoring
            );
            CUDACHECKASYNC;
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Unpacked){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToOne;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToOne,
                SequenceInputDataOneToOne,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
            const int maxNumBlocksByInputSize = (inputData.getNumAlignments() + alignmentsPerBlock - 1) / alignmentsPerBlock;
            const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;

            const int numBlocks = std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy);
            if(numBlocks <= 0){
                throw std::runtime_error("could not launch kernel. numBlocks <= 0");
            }

            kernel<<<numBlocks, blocksize, smem, stream>>>(
                d_scoreOutput,
                inputData,
                gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
                scoring
            );
            CUDACHECKASYNC;
        }
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

        SharedMemorySubstitutionMatrixApproach sharedMemoryApproach 
            = selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_longQuery<SequenceInputDataOneToOne>();

        if(sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Impossible){
            throw std::runtime_error("Could not set shared memory kernel attribute");
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Squared){
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToOne;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_squared_oneToOne,
                SequenceInputDataOneToOne,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumTargetLength);

            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
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
                gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQuerySquared(),
                scoring,
                d_temp,
                tileTempBytesPerGroup
            );
            CUDACHECKASYNC;
        }else if (sharedMemoryApproach == SharedMemorySubstitutionMatrixApproach::Unpacked){
            #if 1
            using SUBMAT = typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToOne;
            auto kernel = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToOne,
                SequenceInputDataOneToOne,
                SUBMAT
            >;

            const int smem = sizeof(SUBMAT);

            int maxBlocksPerSM = 0;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                kernel,
                blocksize, 
                smem
            ));
            const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumTargetLength);

            constexpr int groupsPerBlock = (blocksize / groupsize);
            constexpr int alignmentsPerBlock = 2*groupsPerBlock;
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
                gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(),
                scoring,
                d_temp,
                tileTempBytesPerGroup
            );
            CUDACHECKASYNC;
            #endif
        }
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





    template<
        class InputData
    >
    SharedMemorySubstitutionMatrixApproach selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_shortQuery(){
        static std::map<int, SharedMemorySubstitutionMatrixApproach> approachMap;

        auto it = approachMap.find(deviceId);
        SharedMemorySubstitutionMatrixApproach approach = it == approachMap.end() ? SharedMemorySubstitutionMatrixApproach::NotSelected : it->second;
        if(approach != SharedMemorySubstitutionMatrixApproach::NotSelected) return approach;

        using SUBMAT_squared = typename std::conditional<
            InputData::isSameQueryForAll,
            typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToAll,
            typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToOne
        >::type;
        constexpr SubstitutionMatrixDimensionMode dimMode_squared 
            = InputData::isSameQueryForAll ? KernelSubstitutionMatrixInformation::dimMode_squared_oneToAll : KernelSubstitutionMatrixInformation::dimMode_squared_oneToOne;

        auto kernel_squaredpacked = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            dimMode_squared,
            InputData,
            SUBMAT_squared
        >;
        cudaError_t status = cudaFuncSetAttribute(kernel_squaredpacked, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SUBMAT_squared));
        if(status == cudaSuccess){
            approach = SharedMemorySubstitutionMatrixApproach::Squared;
        }else{
            using SUBMAT_unpacked = typename std::conditional<
                InputData::isSameQueryForAll,
                typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToAll,
                typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToOne
            >::type;
            constexpr SubstitutionMatrixDimensionMode dimMode_SUBMAT_unpacked 
                = InputData::isSameQueryForAll ? KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToAll : KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToOne;

            auto kernel_unpacked = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                dimMode_SUBMAT_unpacked,
                InputData,
                SUBMAT_unpacked
            >;
            status = cudaFuncSetAttribute(kernel_unpacked, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SUBMAT_unpacked));
            if(status == cudaSuccess){
                approach = SharedMemorySubstitutionMatrixApproach::Unpacked;
            }else{
                approach = SharedMemorySubstitutionMatrixApproach::Impossible;
            }
        }
        cudaGetLastError(); //reset error state
        approachMap[deviceId] = approach;
        return approach;
    }

    template<
        class InputData
    >
    SharedMemorySubstitutionMatrixApproach selectSharedMemorySubstitutionMatrixApproachAndSetSmemKernelAttribute_longQuery(){
        static std::map<int, SharedMemorySubstitutionMatrixApproach> approachMap;

        auto it = approachMap.find(deviceId);
        SharedMemorySubstitutionMatrixApproach approach = it == approachMap.end() ? SharedMemorySubstitutionMatrixApproach::NotSelected : it->second;
        if(approach != SharedMemorySubstitutionMatrixApproach::NotSelected) return approach;

        using SUBMAT_squared = typename std::conditional<
            InputData::isSameQueryForAll,
            typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToAll,
            typename KernelSubstitutionMatrixInformation::SUBMAT_squared_oneToOne
        >::type;
        constexpr SubstitutionMatrixDimensionMode dimMode_squared 
            = InputData::isSameQueryForAll ? KernelSubstitutionMatrixInformation::dimMode_squared_oneToAll : KernelSubstitutionMatrixInformation::dimMode_squared_oneToOne;

        auto kernel_squaredpacked = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            dimMode_squared,
            InputData,
            SUBMAT_squared
        >;
        cudaError_t status = cudaFuncSetAttribute(kernel_squaredpacked, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SUBMAT_squared));
        if(status == cudaSuccess){
            approach = SharedMemorySubstitutionMatrixApproach::Squared;
        }else{
            #if 1
            using SUBMAT_unpacked = typename std::conditional<
                InputData::isSameQueryForAll,
                typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToAll,
                typename KernelSubstitutionMatrixInformation::SUBMAT_unpacked_oneToOne
            >::type;
            constexpr SubstitutionMatrixDimensionMode dimMode_SUBMAT_unpacked 
                = InputData::isSameQueryForAll ? KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToAll : KernelSubstitutionMatrixInformation::dimMode_unpacked_oneToOne;

            auto kernel_unpacked = semiglobalalignment::alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
                alphabetSize,
                ScoreType, 
                penaltyType, 
                blocksize, 
                groupsize, 
                numItems,
                dimMode_SUBMAT_unpacked,
                InputData,
                SUBMAT_unpacked
            >;
            status = cudaFuncSetAttribute(kernel_unpacked, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SUBMAT_unpacked));
            if(status == cudaSuccess){
                approach = SharedMemorySubstitutionMatrixApproach::Unpacked;
            }else{
                approach = SharedMemorySubstitutionMatrixApproach::Impossible;
            }
            #else
            approach = SharedMemorySubstitutionMatrixApproach::Impossible;
            #endif
        }
        cudaGetLastError(); //reset error state
        approachMap[deviceId] = approach;
        return approach;
    }

private:
    int deviceId;
    int maxSharedMemoryPerBlockOptin;
    int numSMs;
};










#endif
