#ifndef LOCAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH
#define LOCAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH

#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include "tile_processing.cuh"
#include "state_linear.cuh"
#include "state_affine.cuh"
#include "../substitution_score_provider.cuh"
#include "../util.cuh"
#include "../letter_utilities.cuh"
#include "../cuda_errorcheck.cuh"

#include <map>
#include <iostream>

namespace localalignment{



    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        class InputData,
        class SUBMAT
    >
    __global__
    __launch_bounds__(blocksize,1)
    void alphabet_substitutionmatrix_floatOrInt_kernel(
        __grid_constant__ int* const scoreOutput,
        __grid_constant__ const InputData inputData,
        __grid_constant__ const SUBMAT* const substmatPtr,
        __grid_constant__ const ScoringKernelParam<ScoreType> scoring
    ){
        static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

        static_assert(numItems % 4 == 0);
        static_assert(groupsize <= 32);
        static_assert(blocksize % groupsize == 0);

        //require extra space for "out-of-bounds element"
        constexpr int expectedNumColumnsSUBMAT = alphabetSize+1;
        constexpr int expectedNumRowsSUBMAT = alphabetSize+1;

        static_assert(expectedNumRowsSUBMAT == SUBMAT::numRows);
        static_assert(expectedNumColumnsSUBMAT == SUBMAT::numColumns);
        
        __builtin_assume(blockDim.x == blocksize);
        __builtin_assume(blockDim.x % groupsize == 0);
        __builtin_assume(groupsize <= 32);

        auto group = cooperative_groups::tiled_partition<groupsize>(cooperative_groups::this_thread_block());
        
        const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
    
        constexpr int paddingLetter = alphabetSize;
        constexpr int relaxChunkSize = 4;

        using MathOps = MathOps<ScoreType>;
        using UpdateMaxOp = UpdateMax<ScoreType>;
        using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LocalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
            LocalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
        >::type;
        using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
        
        extern __shared__ float4 externalSharedMem[];
        SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

        for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
            const int row = i / SUBMAT::numColumns;
            const int col = i % SUBMAT::numColumns;
            shared_substmat.data[row][col] = substmatPtr->data[row][col];
        }
        __syncthreads();

        int queryLetters[numItems];
        int queryLength = 0;
        //load query outside of loop in case of single query
        if constexpr(InputData::isSameQueryForAll){
            const auto* query = inputData.getQuery(0);
            queryLength = inputData.getQueryLength(0);
            #pragma unroll
            for (int i=0; i < numItems; i++) {
                if (numItems * group.thread_rank() + i >= queryLength) queryLetters[i] = paddingLetter;
                else queryLetters[i] = query[numItems * group.thread_rank()+i]; 
            }
        }

        SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);


        for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){
            //load query inside of loop in case of multi-query
            if constexpr(!InputData::isSameQueryForAll){
                const auto* query = inputData.getQuery(alignmentId);
                queryLength = inputData.getQueryLength(alignmentId);
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    if (numItems * group.thread_rank() + i >= queryLength) queryLetters[i] = paddingLetter;
                    else queryLetters[i] = query[numItems * group.thread_rank()+i]; 
                }
            }

            UpdateMaxOp maximumTracker;
            State state(substitutionProvider, group, maximumTracker, scoring);

            const int subjectLength = inputData.getSubjectLength(alignmentId);
            const std::int8_t* const subjectData = inputData.getSubject(alignmentId);
            SubjectLettersData subjectLetters(group, subjectData, subjectLength);     
    
            subjectLetters.loadNext4Letters();
            state.initScores(0, FirstLeftBorder<ScoreType>{});
    
            const int outputThreadRank = (queryLength-1) / numItems;
            const int numRows = subjectLength + outputThreadRank + 1;

            processSingleTile(
                group,
                state,
                subjectLetters,
                1,
                numRows
            ); 

            const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
            if(group.thread_rank() == 0){
                scoreOutput[alignmentId] = groupmax;
            }
        }
    }



    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        class InputData,
        class SUBMAT
    >
    void call_alphabet_substitutionmatrix_floatOrInt_kernel(
        int* d_scoreOutput,
        const InputData& inputData,
        const SUBMAT* d_substmatPtr,
        const ScoringKernelParam<ScoreType>& scoring,
        char* /*d_temp*/, //must be aligned to 256 bytes
        size_t /*tempBytes*/,
        cudaStream_t stream
    ){
        // if(((size_t)d_temp) % 256 != 0){
        //     throw std::runtime_error("d_temp not aligned to 256 bytes");
        // }
        auto kernel = alphabet_substitutionmatrix_floatOrInt_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            InputData,
            SUBMAT
        >;

        int smem = sizeof(SUBMAT);

        auto setSmemKernelAttribute = [&](){
            static std::map<int, bool> isSet;
            if(smem > 48*1024){
                int deviceId = 0;
                cudaGetDevice(&deviceId);
                if(!isSet[deviceId]){
                    int availableSmem = 0;
                    cudaDeviceGetAttribute(&availableSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
                    // if(smem > availableSmem) throw std::runtime_error("too much shared memory required");
                    if(smem > availableSmem) return false;
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                    isSet[deviceId] = true;
                }
            }
            return true;
        };

        bool smemOk = setSmemKernelAttribute();
        if(!smemOk){
            std::cout << "Not enough smem available. Setting scores to 0";
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments(), stream));
            return;
        }

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
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
        //no temp usage
        const int maxNumBlocksByTempBytes = maxNumBlocksByOccupancy; //tempBytes / (sizeof(ScoreType) * groupsPerBlock * numItems);

        const int numBlocks = std::min(maxNumBlocksByTempBytes, std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy));
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            d_substmatPtr,
            scoring
        );
        CUDACHECKASYNC;
    }




    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        SubstitutionMatrixDimensionMode substmatDimMode,
        class InputData,
        class SUBMAT
    >
    __global__
    __launch_bounds__(blocksize,1)
    void alphabet_substitutionmatrix_half2OrShort2_kernel(
        __grid_constant__ int* const scoreOutput,
        __grid_constant__ const InputData inputData,
        __grid_constant__ const SUBMAT* const substmatPtr,
        __grid_constant__ const ScoringKernelParam<ScoreType> scoring
    ){
        static_assert(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>);

        static_assert(numItems % 4 == 0);
        static_assert(groupsize <= 32);
        static_assert(blocksize % groupsize == 0);
        /*        
            single query:
                packed : [dim*dim][dim] half2
                all unpacked : [dim][dim] half

            multi query:
                packed : [dim*dim][dim*dim] half2
                subject unpacked : [dim][dim*dim] half2
                query unpacked : [dim*dim][dim] half2
                all unpacked : [dim][dim] half
        */

        if constexpr (InputData::isSameQueryForAll){
            static_assert(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear);
        }else{
            static_assert(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear);
        }

        #if 1

        static_assert(
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared)
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
                static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1))))
            &&
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
                static_assert(SUBMAT::numColumns == (alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
                    static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize+1));
                    static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
                    static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize+1));
                    static_assert(SUBMAT::numColumns == (alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear) 
                || ((std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1))))

        );

        #else

        if constexpr(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
            static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
            static_assert(SUBMAT::numColumns == (alphabetSize+1));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize+1));
            static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
            static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize+1));
            static_assert(SUBMAT::numColumns == (alphabetSize+1));
        }

        #endif

        //do not compile half2 code for gpus without native half2 arithmetic instructions.
        #if __CUDA_ARCH__ < 800
        #ifndef COMPILE_HALF2_FOR_ARCHS_WITHOUT_HALF2_MATH
        if constexpr(std::is_same_v<ScoreType, half2>){
            for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < inputData.getNumAlignments(); i += blockDim.x * gridDim.x){
                scoreOutput[i] = 0;
            }
            return;
        }
        #endif
        #endif

        //do not compile short2 code for gpus without dpx instructions.
        #if __CUDA_ARCH__ < 900
        #ifndef COMPILE_SHORT2_FOR_ARCHS_WITHOUT_DPX
        if constexpr(std::is_same_v<ScoreType, short2>){
            for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < inputData.getNumAlignments(); i += blockDim.x * gridDim.x){
                scoreOutput[i] = 0;
            }
            return;
        }
        #endif
        #endif
        
        __builtin_assume(blockDim.x == blocksize);
        __builtin_assume(blockDim.x % groupsize == 0);
        __builtin_assume(groupsize <= 32);
        

        auto group = cooperative_groups::tiled_partition<groupsize>(cooperative_groups::this_thread_block());
        
        const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
    
        constexpr int paddingLetter = alphabetSize;
        constexpr int relaxChunkSize = 4;

        using SubstitutionScoreProviderSameQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, false> // SubjectLinearQueryLinear
        >::type;

        using SubstitutionScoreProviderDifferentQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            typename std::conditional<
                substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
                SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), false, true>,
                typename std::conditional<
                    substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, false>,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, true> //SubjectLinearQueryLinear
                >::type
            >::type
        >::type;

        using SubstitutionScoreProvider = typename std::conditional<
            InputData::isSameQueryForAll,
            SubstitutionScoreProviderSameQuery,
            SubstitutionScoreProviderDifferentQuery
        >::type;

        using MathOps = MathOps<ScoreType>;
        using UpdateMaxOp = UpdateMax<ScoreType>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LocalAlignmentLinearGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
            LocalAlignmentAffineGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
        >::type;
        using SubjectPairLettersData = SubjectPairLettersData<alphabetSize+1, decltype(group)>;
        
        extern __shared__ float4 externalSharedMem[];
        SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

        for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
            const int row = i / SUBMAT::numColumns;
            const int col = i % SUBMAT::numColumns;
            shared_substmat.data[row][col] = substmatPtr->data[row][col];
        }
        __syncthreads();

        int queryLetters[numItems];
        int queryLength0 = 0;
        int queryLength1 = 0;
        //load query outside of loop in case of single query
        if constexpr(InputData::isSameQueryForAll){
            const auto* query = inputData.getQuery(0);
            queryLength0 = inputData.getQueryLength(0);
            queryLength1 = queryLength0;
            #pragma unroll
            for (int i=0; i < numItems; i++) {
                if (numItems * group.thread_rank() + i >= queryLength0) queryLetters[i] = paddingLetter;
                else queryLetters[i] = query[numItems * group.thread_rank()+i]; 
            }
        }


        SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

        for(int alignmentId = 2*groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid*2){
            const int alignmentId0 = alignmentId + 0;
            const int alignmentId1 = alignmentId + 1;

            //load queries inside of loop in case of multi-query
            if constexpr(!InputData::isSameQueryForAll){
                queryLength0 = inputData.getQueryLength(alignmentId0);
                const std::int8_t* const query0 = inputData.getQuery(alignmentId0);

                queryLength1 = queryLength0;
                const std::int8_t* query1 = query0;
                if(alignmentId1 < inputData.getNumAlignments()){
                    queryLength1 = inputData.getQueryLength(alignmentId1);
                    query1 = inputData.getQuery(alignmentId1);
                }

                //load query
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    int templetter0;
                    if (numItems * group.thread_rank() + i >= queryLength0) templetter0 = paddingLetter;
                    else templetter0 = query0[numItems * group.thread_rank()+i]; 
                    int templetter1;
                    if (numItems * group.thread_rank() + i >= queryLength1) templetter1 = paddingLetter;
                    else templetter1 = query1[numItems * group.thread_rank()+i]; 

                    queryLetters[i] = FuseTwoEncodedLetters{}.single(templetter0, templetter1, alphabetSize+1);
                }
            }

            UpdateMaxOp maximumTracker;
            State state(substitutionProvider, group, maximumTracker, scoring);


            const int subjectLength0 = inputData.getSubjectLength(alignmentId0);
            const std::int8_t* const subjectData0 = inputData.getSubject(alignmentId0);

            int subjectLength1 = subjectLength0;
            const std::int8_t* subjectData1 = subjectData0;
            if(alignmentId1 < inputData.getNumAlignments()){
                subjectLength1 = inputData.getSubjectLength(alignmentId1);
                subjectData1 = inputData.getSubject(alignmentId1);
            }

            SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);
    
            subjectLetters.loadNext4Letters();
            state.initScores(0, FirstLeftBorder<ScoreType>{});
            const int outputThreadRank0 = (queryLength0-1) / numItems;
            const int outputThreadRank1 = (queryLength1-1) / numItems;
            if constexpr(InputData::isSameQueryForAll){
                __builtin_assume(outputThreadRank0 == outputThreadRank1);
            }
            const int numRows0 = subjectLength0 + outputThreadRank0 + 1;
            const int numRows1 = subjectLength1 + outputThreadRank1 + 1;
            const int numRows = max(numRows0, numRows1);

            processSingleTile(
                group,
                state,
                subjectLetters,
                1,
                numRows
            );

            const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
            if(group.thread_rank() == 0){
                scoreOutput[alignmentId0] = groupmax.x;
            }
            if(alignmentId1 < inputData.getNumAlignments()){
                if(group.thread_rank() == 0){
                    scoreOutput[alignmentId1] = groupmax.y;
                }
            }
        }
    }



    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        SubstitutionMatrixDimensionMode substmatDimMode,
        class InputData,
        class SUBMAT
    >
    void call_alphabet_substitutionmatrix_half2OrShort2_kernel(
        int* d_scoreOutput,
        const InputData& inputData,
        const SUBMAT* d_substmatPtr,
        const ScoringKernelParam<ScoreType>& scoring,
        char* /*d_temp*/, //must be aligned to 256 bytes
        size_t /*tempBytes*/,
        cudaStream_t stream
    ){
        // if(((size_t)d_temp) % 256 != 0){
        //     throw std::runtime_error("d_temp not aligned to 256 bytes");
        // }
        auto kernel = alphabet_substitutionmatrix_half2OrShort2_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            substmatDimMode,
            InputData,
            SUBMAT
        >;

        int smem = sizeof(SUBMAT);

        auto setSmemKernelAttribute = [&](){
            static std::map<int, bool> isSet;
            if(smem > 48*1024){
                int deviceId = 0;
                cudaGetDevice(&deviceId);
                if(!isSet[deviceId]){
                    int availableSmem = 0;
                    cudaDeviceGetAttribute(&availableSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
                    // if(smem > availableSmem) throw std::runtime_error("too much shared memory required");
                    if(smem > availableSmem) return false;
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                    isSet[deviceId] = true;
                }
            }
            return true;
        };

        bool smemOk = setSmemKernelAttribute();
        if(!smemOk){
            std::cout << "Not enough smem available. Setting scores to 0";
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments(), stream));
            return;
        }

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
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
        //no temp usage
        const int maxNumBlocksByTempBytes = maxNumBlocksByOccupancy; //tempBytes / (sizeof(ScoreType) * groupsPerBlock * numItems);

        const int numBlocks = std::min(maxNumBlocksByTempBytes, std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy));
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        kernel<<<numBlocks, blocksize, smem, stream>>>(
            d_scoreOutput,
            inputData,
            d_substmatPtr,
            scoring
        );
        CUDACHECKASYNC;
    }







    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        class InputData,
        class SUBMAT
    >
    __global__
    __launch_bounds__(blocksize,1)
    void alphabet_substitutionmatrix_floatOrInt_multipass_kernel(
        __grid_constant__ int* const scoreOutput,
        __grid_constant__ const InputData inputData,
        __grid_constant__ const SUBMAT* const substmatPtr,
        __grid_constant__ const ScoringKernelParam<ScoreType> scoring,
        __grid_constant__ char* const tempStorage,
        __grid_constant__ const size_t tempBytesPerGroup
    ){      
        static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

        //groupsize >= 4 because we process 4 chars in one go and write to temp with the group once afterwards
        static_assert(groupsize >= 4);

        static_assert(numItems % 4 == 0);
        static_assert(groupsize <= 32);
        static_assert(blocksize % groupsize == 0);

        constexpr int expectedNumColumnsSUBMAT = alphabetSize+1;
        constexpr int expectedNumRowsSUBMAT = alphabetSize+1;

        static_assert(expectedNumRowsSUBMAT == SUBMAT::numRows);
        static_assert(expectedNumColumnsSUBMAT == SUBMAT::numColumns);

        __builtin_assume(blockDim.x == blocksize);
        __builtin_assume(blockDim.x % groupsize == 0);
        __builtin_assume(groupsize >= 4);
        __builtin_assume(groupsize <= 32);
        
        auto group = cooperative_groups::tiled_partition<groupsize>(cooperative_groups::this_thread_block());
        
        const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
        // constexpr int numGroupsPerBlock = blocksize / groupsize;
        // constexpr int tileSize = groupsize * numItems;

        constexpr int paddingLetter = alphabetSize;
        constexpr int relaxChunkSize = 4;

        extern __shared__ float4 externalSharedMem[];
        SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

        for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
            const int row = i / SUBMAT::numColumns;
            const int col = i % SUBMAT::numColumns;
            shared_substmat.data[row][col] = substmatPtr->data[row][col];
        }
        __syncthreads();

        using MathOps = MathOps<ScoreType>;
        using UpdateMaxOp = UpdateMax<ScoreType>;
        using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
        using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LocalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
            LocalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
        >::type;

        using LeftBorderType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LeftBorderLinear<ScoreType>,
            LeftBorderAffine<ScoreType>
        >::type;
        using TempStorageDataType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            ScoreType,
            typename Vectorized2<ScoreType>::type
        >::type;
        using TempHandler = TempHandler<decltype(group), TempStorageDataType>;
        using LastColumnInLastThread = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LastColumnLinearLastThread<ScoreType>,
            LastColumnAffineLastThread<ScoreType>
        >::type;



        TempStorageDataType* const groupTempStorage = (TempStorageDataType*)(((char*)tempStorage) + tempBytesPerGroup * groupIdInGrid);

        auto clearOutOfTileTempStorage = [&](int subjectLength){
            if(group.thread_rank() < group.size() - 1){
                groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
            }
        };

        for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){

            const auto* query = inputData.getQuery(alignmentId);
            const int queryLength = inputData.getQueryLength(alignmentId);
            const int numTiles = SDIV(queryLength, groupsize * numItems);

            auto loadQueryLetters = [&](int tileNr, int (&queryLetters)[numItems]){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    const int index = tileNr * groupsize * numItems + numItems * group.thread_rank()+i;
                    if (index >= queryLength) queryLetters[i] = paddingLetter;
                    else queryLetters[i] = query[index]; 
                }
            };
            if(numTiles == 1){
                int queryLetters[numItems];
                loadQueryLetters(0, queryLetters);
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                UpdateMaxOp maximumTracker;
                State state(substitutionProvider, group, maximumTracker, scoring);

                const int subjectLength = inputData.getSubjectLength(alignmentId);
                const std::int8_t* const subjectData = inputData.getSubject(alignmentId);
                SubjectLettersData subjectLetters(group, subjectData, subjectLength);        
        
                subjectLetters.loadNext4Letters();
                state.initScores(0, FirstLeftBorder<ScoreType>{});
        
                const int outputThreadRank = (queryLength-1) / numItems;
                const int numRows = subjectLength + outputThreadRank + 1;

                processSingleTile(
                    group,
                    state,
                    subjectLetters,
                    1,
                    numRows
                );    
                
                const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
                if(group.thread_rank() == 0){
                    scoreOutput[alignmentId] = groupmax;
                }
            }else{
                UpdateMaxOp maximumTracker;
                LeftBorderType leftBorder;

                int subjectLength = 0;
                const std::int8_t* subjectData = nullptr;


                /* 
                    -----------------------
                    Process tile 0
                    ----------------------- 
                */
                {

                    int queryLetters[numItems];
                    loadQueryLetters(0, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    State state(substitutionProvider, group, maximumTracker, scoring);

                    subjectLength = inputData.getSubjectLength(alignmentId);
                    subjectData = inputData.getSubject(alignmentId);
                    clearOutOfTileTempStorage(subjectLength);

                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);
                    subjectLetters.loadNext4Letters();

                    LastColumnInLastThread lastColumn;

                    state.initScores(0, FirstLeftBorder<ScoreType>{});

                    const int numRows = subjectLength + (group.size()-1) + 1;

                    TempHandler tempHandler(group, groupTempStorage);
                    processFirstTile(
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        subjectLength,
                        lastColumn,
                        tempHandler
                    );

                }

                //process intermediate tiles
                for(int tileNr = 1; tileNr < numTiles-1; tileNr++){
                    /* 
                        -----------------------
                        Process tile tileNr
                        ----------------------- 
                    */
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    State state(substitutionProvider, group, maximumTracker, scoring);
                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);
                    const int numRows = subjectLength + (group.size()-1) + 1;

                    processIntermediateTile(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        subjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                }

                //process last tile
                {
                    const int tileNr = numTiles-1;
                    /* 
                        -----------------------
                        Process tile numTiles-1
                        ----------------------- 
                    */

                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    State state(substitutionProvider, group, maximumTracker, scoring);
                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    state.initScores(tileNr, leftBorder);

                    const int queryLengthInLastTile = queryLength - (numTiles-1) * (groupsize * numItems);
                    const int outputThreadRank = (queryLengthInLastTile-1) / numItems;
                    const int numRows = subjectLength + outputThreadRank + 1;

                    processLastTile(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        1,
                        numRows,
                        subjectLength,
                        leftBorder,
                        tempHandler
                    );

                    const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
                    if(group.thread_rank() == 0){
                        scoreOutput[alignmentId] = groupmax;
                    }
                }

            }

        }

    }

    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        class InputData,
        class SUBMAT
    >
    void call_alphabet_substitutionmatrix_floatOrInt_multipass_kernel(
        int* d_scoreOutput,
        const InputData& inputData,
        const SUBMAT* d_substmatPtr,
        const ScoringKernelParam<ScoreType>& scoring,
        int maxSubjectLength,
        char* d_temp, //must be aligned to 256 bytes
        size_t tempBytes,
        cudaStream_t stream
    ){
        if(((size_t)d_temp) % 256 != 0){
            throw std::runtime_error("d_temp not aligned to 256 bytes");
        }
        auto kernel = alphabet_substitutionmatrix_floatOrInt_multipass_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            InputData,
            SUBMAT
        >;
    
        int smem = sizeof(SUBMAT);

        auto setSmemKernelAttribute = [&](){
            static std::map<int, bool> isSet;
            if(smem > 48*1024){
                int deviceId = 0;
                cudaGetDevice(&deviceId);
                if(!isSet[deviceId]){
                    int availableSmem = 0;
                    cudaDeviceGetAttribute(&availableSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
                    // if(smem > availableSmem) throw std::runtime_error("too much shared memory required");
                    if(smem > availableSmem) return false;
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                    isSet[deviceId] = true;
                }
            }
            return true;
        };

        bool smemOk = setSmemKernelAttribute();
        if(!smemOk){
            std::cout << "Not enough smem available. Setting scores to 0";
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments(), stream));
            return;
        }

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));

        using TempStorageDataType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            ScoreType,
            typename Vectorized2<ScoreType>::type
        >::type;
        const int maxSubjectLengthPadded = maxSubjectLength + groupsize;
        const size_t tileTempBytesPerGroup = sizeof(TempStorageDataType) * maxSubjectLengthPadded;

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
            d_substmatPtr,
            scoring,
            d_temp,
            tileTempBytesPerGroup
        );
        CUDACHECKASYNC;
    }

    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        SubstitutionMatrixDimensionMode substmatDimMode,
        class InputData,
        class SUBMAT
    >
    __global__
    __launch_bounds__(blocksize,1)
    void alphabet_substitutionmatrix_half2OrShort2_multipass_kernel(
        __grid_constant__ int* const scoreOutput,
        __grid_constant__ const InputData inputData,
        __grid_constant__ const SUBMAT* const substmatPtr,
        __grid_constant__ const ScoringKernelParam<ScoreType> scoring,
        __grid_constant__ char* const tempStorage,
        __grid_constant__ const size_t tempBytesPerGroup
    ){

        static_assert(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>);
        //groupsize >= 4 because we process 4 chars in one go and write to temp with the group once afterwards
        static_assert(groupsize >= 4);

        static_assert(numItems % 4 == 0);
        static_assert(groupsize <= 32);
        static_assert(blocksize % groupsize == 0);

        /*        
            single query:
                packed : [dim*dim][dim] half2
                all unpacked : [dim][dim] half

            multi query:
                packed : [dim*dim][dim*dim] half2
                subject unpacked : [dim][dim*dim] half2
                query unpacked : [dim*dim][dim] half2
                all unpacked : [dim][dim] half
        */

        if constexpr (InputData::isSameQueryForAll){
            static_assert(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear);
        }else{
            static_assert(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared
                || substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear);
        }


        static_assert(
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared)
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
                static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1))))
            &&
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1));
                static_assert(SUBMAT::numColumns == (alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)*(alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
                    static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize+1));
                    static_assert(SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1)*(alphabetSize+1))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
                    static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize+1));
                    static_assert(SUBMAT::numColumns == (alphabetSize+1));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear) 
                || ((std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize+1)) 
                    && (SUBMAT::numColumns == (alphabetSize+1))))

        );

        //do not compile half2 code for gpus without native half2 arithmetic instructions.
        #if __CUDA_ARCH__ < 800
        #ifndef COMPILE_HALF2_FOR_ARCHS_WITHOUT_HALF2_MATH
        if constexpr(std::is_same_v<ScoreType, half2>){
            for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < inputData.getNumAlignments(); i += blockDim.x * gridDim.x){
                scoreOutput[i] = 0;
            }
            return;
        }
        #endif
        #endif

        //do not compile short2 code for gpus without dpx instructions.
        #if __CUDA_ARCH__ < 900
        #ifndef COMPILE_SHORT2_FOR_ARCHS_WITHOUT_DPX
        if constexpr(std::is_same_v<ScoreType, short2>){
            for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < inputData.getNumAlignments(); i += blockDim.x * gridDim.x){
                scoreOutput[i] = 0;
            }
            return;
        }
        #endif
        #endif

        __builtin_assume(blockDim.x == blocksize);
        __builtin_assume(blockDim.x % groupsize == 0);
        __builtin_assume(groupsize >= 4);
        __builtin_assume(groupsize <= 32);
        
        auto group = cooperative_groups::tiled_partition<groupsize>(cooperative_groups::this_thread_block());
        
        const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
        // constexpr int numGroupsPerBlock = blocksize / groupsize;
        // constexpr int tileSize = groupsize * numItems;

        constexpr int paddingLetter = alphabetSize;

        using SubstitutionScoreProviderSameQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, false> // SubjectLinearQueryLinear
        >::type;

        using SubstitutionScoreProviderDifferentQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            typename std::conditional<
                substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
                SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), false, true>,
                typename std::conditional<
                    substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, false>,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize+1), true, true> //SubjectLinearQueryLinear
                >::type
            >::type
        >::type;

        using SubstitutionScoreProvider = typename std::conditional<
            InputData::isSameQueryForAll,
            SubstitutionScoreProviderSameQuery,
            SubstitutionScoreProviderDifferentQuery
        >::type;

        using MathOps = MathOps<ScoreType>;
        using UpdateMaxOp = UpdateMax<ScoreType>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LocalAlignmentLinearGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp>,
            LocalAlignmentAffineGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp>
        >::type;
        using SubjectPairLettersData = SubjectPairLettersData<alphabetSize+1, decltype(group)>;
        
        using LeftBorderType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LeftBorderLinear<ScoreType>,
            LeftBorderAffine<ScoreType>
        >::type;
        using TempStorageDataType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            ScoreType,
            typename Vectorized2<ScoreType>::type
        >::type;
        using TempHandler = TempHandler<decltype(group), TempStorageDataType>;
        using LastColumnInLastThread = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            LastColumnLinearLastThread<ScoreType>,
            LastColumnAffineLastThread<ScoreType>
        >::type;
        
        extern __shared__ float4 externalSharedMem[];
        SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

        for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
            const int row = i / SUBMAT::numColumns;
            const int col = i % SUBMAT::numColumns;
            shared_substmat.data[row][col] = substmatPtr->data[row][col];
        }
        __syncthreads();

        TempStorageDataType* const groupTempStorage = (TempStorageDataType*)(((char*)tempStorage) + tempBytesPerGroup * groupIdInGrid);

        auto clearOutOfTileTempStorage = [&](int subjectLength){
            if(group.thread_rank() < group.size() - 1){
                groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
            }
        };

        for(int alignmentId = 2*groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid*2){
            const int alignmentId0 = alignmentId + 0;
            const int alignmentId1 = alignmentId + 1;

            const int subjectLength0 = inputData.getSubjectLength(alignmentId0);
            int subjectLength1 = subjectLength0;
            const std::int8_t* const subjectData0 = inputData.getSubject(alignmentId0);
            const std::int8_t* subjectData1 = subjectData0;
            if(alignmentId1 < inputData.getNumAlignments()){
                subjectLength1 = inputData.getSubjectLength(alignmentId1);
                subjectData1 = inputData.getSubject(alignmentId1);
            }

            const int queryLength0 = inputData.getQueryLength(alignmentId0);
            int queryLength1 = queryLength0;
            const std::int8_t* const query0 = inputData.getQuery(alignmentId0);
            const std::int8_t* query1 = query0;

            if(alignmentId1 < inputData.getNumAlignments()){
                queryLength1 = inputData.getQueryLength(alignmentId1);
                query1 = inputData.getQuery(alignmentId1);
            }
            if constexpr(InputData::isSameQueryForAll){
                __builtin_assume(queryLength0 == queryLength1);
                __builtin_assume(query0 == query1);
            }

            auto loadQueryLetters = [&](int tileNr, int (&queryLetters)[numItems]){
                const int positionOffset = tileNr * groupsize * numItems;

                if constexpr(InputData::isSameQueryForAll){
                    #pragma unroll
                    for (int i=0; i < numItems; i++) {
                        const int index = positionOffset + numItems * group.thread_rank()+i;
                        if (index >= queryLength0) queryLetters[i] = paddingLetter;
                        else queryLetters[i] = query0[index]; 
                    }
                }else{
                    #pragma unroll
                    for (int i=0; i < numItems; i++) {
                        const int index = positionOffset + numItems * group.thread_rank()+i;

                        int templetter0;
                        if (index >= queryLength0) templetter0 = paddingLetter;
                        else templetter0 = query0[index]; 
                        int templetter1;
                        if (index >= queryLength1) templetter1 = paddingLetter;
                        else templetter1 = query1[index]; 

                        queryLetters[i] = FuseTwoEncodedLetters{}.single(templetter0, templetter1, alphabetSize+1);
                    }
                }
                
            };

            const int largerQueryLength = max(queryLength0, queryLength1);
            const int largerSubjectLength = max(subjectLength0, subjectLength1);
            const int numTiles = SDIV(largerQueryLength, groupsize * numItems);
        
            if(numTiles == 1){
                int queryLetters[numItems];
                loadQueryLetters(0, queryLetters);
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                UpdateMaxOp maximumTracker;
                State state(substitutionProvider, group, maximumTracker, scoring);
                SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);
        
                subjectLetters.loadNext4Letters();
                state.initScores(0, FirstLeftBorder<ScoreType>{});

                const int lastColThreadRank = (largerQueryLength-1) / numItems;
                const int numRows = largerSubjectLength + lastColThreadRank + 1;

                processSingleTile(
                    group,
                    state,
                    subjectLetters,
                    1,
                    numRows
                );

                const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
                if(group.thread_rank() == 0){
                    scoreOutput[alignmentId0] = groupmax.x;
                }
                if(alignmentId1 < inputData.getNumAlignments()){
                    if(group.thread_rank() == 0){
                        scoreOutput[alignmentId1] = groupmax.y;
                    }
                }
            }else{
                UpdateMaxOp maximumTracker;
                LeftBorderType leftBorder;

                {
                    /* 
                        -----------------------
                        Process tile 0
                        ----------------------- 
                    */

                    clearOutOfTileTempStorage(largerSubjectLength);
                    int queryLetters[numItems];
                    loadQueryLetters(0, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    State state(substitutionProvider, group, maximumTracker, scoring);
                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();
                    state.initScores(0, FirstLeftBorder<ScoreType>{});

                    LastColumnInLastThread lastColumn;

                    const int numRows = largerSubjectLength + (group.size()-1) + 1;

                    TempHandler tempHandler(group, groupTempStorage);
                    processFirstTile(
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        largerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                }

                //process intermediate tiles
                for(int tileNr = 1; tileNr < numTiles-1; tileNr++){
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    State state(substitutionProvider, group, maximumTracker, scoring);
                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);

                    const int numRows = largerSubjectLength + (group.size()-1) + 1;

                    processIntermediateTile(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        largerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                }

                //process last tile
                {
                    const int tileNr = numTiles-1;
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);
                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    State state(substitutionProvider, group, maximumTracker, scoring);
                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    state.initScores(tileNr, leftBorder);

                    const int queryLengthInLastTile = largerQueryLength - (numTiles-1) * (groupsize * numItems);
                    const int lastColThreadRank = (queryLengthInLastTile-1) / numItems;
                    const int numRows = largerSubjectLength + lastColThreadRank + 1;

                    processLastTile(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        1,
                        numRows,
                        largerSubjectLength,
                        leftBorder,
                        tempHandler
                    );

                    const ScoreType groupmax = MathOps::reduce_max(group, maximumTracker.maximum);
                    if(group.thread_rank() == 0){
                        scoreOutput[alignmentId0] = groupmax.x;
                    }
                    if(alignmentId1 < inputData.getNumAlignments()){
                        if(group.thread_rank() == 0){
                            scoreOutput[alignmentId1] = groupmax.y;
                        }
                    }
                }
            }

        }
    }



    template<
        int alphabetSize,
        class ScoreType, 
        PenaltyType penaltyType, 
        int blocksize, 
        int groupsize, 
        int numItems,
        SubstitutionMatrixDimensionMode substmatDimMode,
        class SUBMAT,
        class InputData
    >
    void call_alphabet_substitutionmatrix_half2OrShort2_multipass_kernel(
        int* d_scoreOutput,
        const InputData& inputData,
        const SUBMAT& substmat,
        const ScoringKernelParam<ScoreType>& scoring,
        int maxSubjectLength,
        char* d_temp, //must be aligned to 256 bytes
        size_t tempBytes,
        cudaStream_t stream
    ){
        if(((size_t)d_temp) % 256 != 0){
            throw std::runtime_error("d_temp not aligned to 256 bytes");
        }
        auto kernel = alphabet_substitutionmatrix_half2OrShort2_multipass_kernel<
            alphabetSize,
            ScoreType, 
            penaltyType, 
            blocksize, 
            groupsize, 
            numItems,
            substmatDimMode,
            SUBMAT,
            InputData
        >;
    
        int smem = sizeof(SUBMAT);

        auto setSmemKernelAttribute = [&](){
            static std::map<int, bool> isSet;
            if(smem > 48*1024){
                int deviceId = 0;
                cudaGetDevice(&deviceId);
                if(!isSet[deviceId]){
                    int availableSmem = 0;
                    cudaDeviceGetAttribute(&availableSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
                    // if(smem > availableSmem) throw std::runtime_error("too much shared memory required");
                    if(smem > availableSmem) return false;
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                    isSet[deviceId] = true;
                }
            }
            return true;
        };

        bool smemOk = setSmemKernelAttribute();
        if(!smemOk){
            std::cout << "Not enough smem available. Setting scores to 0";
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments(), stream));
            return;
        }

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));

        using TempStorageDataType = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            ScoreType,
            typename Vectorized2<ScoreType>::type
        >::type;
        const int maxSubjectLengthPadded = maxSubjectLength + groupsize;
        const size_t tileTempBytesPerGroup = sizeof(TempStorageDataType) * maxSubjectLengthPadded;

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
            substmat,
            scoring,
            d_temp,
            tileTempBytesPerGroup
        );
        CUDACHECKASYNC;
    }

    




} //namespace localalignment







#endif