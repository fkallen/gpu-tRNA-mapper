#ifndef SEMIGLOBAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH
#define SEMIGLOBAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH

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

namespace semiglobalalignment{

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

        constexpr int expectedNumColumnsSUBMAT = alphabetSize;
        constexpr int expectedNumRowsSUBMAT = alphabetSize;

        static_assert(expectedNumRowsSUBMAT == SUBMAT::numRows);
        static_assert(expectedNumColumnsSUBMAT == SUBMAT::numColumns);
        
        __builtin_assume(blockDim.x == blocksize);
        __builtin_assume(blockDim.x % groupsize == 0);
        __builtin_assume(groupsize <= 32);
        
        auto group = cooperative_groups::tiled_partition<groupsize>(cooperative_groups::this_thread_block());
        
        const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
    
        constexpr int paddingLetter = alphabetSize-1;
        constexpr int relaxChunkSize = 4;

        using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
        using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;

        using MathOps = MathOps<ScoreType>;
        using UpdateMaxInLastColumnOp = UpdateMaxInLastColumn<ScoreType, numItems>;
        using UpdateMaxInLastRowOp = UpdateMaxInLastRow<ScoreType>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>
        >::type;
 
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

            UpdateMaxInLastColumnOp maximumInLastColumnTracker;
            UpdateMaxInLastRowOp maximumInLastRowTracker;

            State state(substitutionProvider, group, maximumInLastColumnTracker, scoring);
            const int subjectLength = inputData.getSubjectLength(alignmentId);
            const std::int8_t* const subjectData = inputData.getSubject(alignmentId);
            SubjectLettersData subjectLetters(group, subjectData, subjectLength);        
    
            subjectLetters.loadNext4Letters();
            state.initScores(0, FirstLeftBorder<ScoreType>{});

            constexpr int tileNr = 0;

            auto trackLastRow = [&](int row){
                if(row - group.thread_rank() == subjectLength){
                    const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        const int position = positionOffset + i;
                        if(position < queryLength){
                            maximumInLastRowTracker(state.scoresM[i], tileNr, i);
                        }
                    }            
                }
            };

            auto doNotTrackLastRow = [](int /*row*/){};
    
            #if 1
            const int outputThreadRank = (queryLength-1) / numItems;
            const int numRows = subjectLength + outputThreadRank + 1;

            processSingleTile_fromStart(
                group,
                state,
                subjectLetters,
                numRows,
                trackLastRow
            ); 

            #else

            const int outputThreadRank = (queryLength-1) / numItems;
            const int numRowsInDPMatrix = subjectLength + outputThreadRank + 1;
            const int numRowsWithoutLastRowForAny = subjectLength-1;

            int r = processSingleTile_alwayscheck_fromStart(
                group,
                state,
                subjectLetters,
                numRowsWithoutLastRowForAny,
                doNotTrackLastRow
            );
            processSingleTile_alwayscheck_continued(
                group,
                state,
                subjectLetters,
                r,
                numRowsInDPMatrix,
                trackLastRow
            );

            #endif

            // printf("tid %d row max %f\n", threadIdx.x, maximumInLastRowTracker.maximum);
            const ScoreType groupmaxLastRow = MathOps::reduce_max(group, maximumInLastRowTracker.maximum);            
            const int outputRegIndex = (queryLength-1) % numItems;
            if(group.thread_rank() == outputThreadRank){
                ScoreType temp[numItems];
                #pragma unroll
                for(int i = 0; i < numItems; i++){
                    temp[i] = maximumInLastColumnTracker.maxima[i];
                }

                const ScoreType maxLastColumn = temp[outputRegIndex];
                // printf("groupmaxLastRow %f, maxLastColumn %f\n", groupmaxLastRow, maxLastColumn);
                scoreOutput[alignmentId] = MathOps::max(groupmaxLastRow, maxLastColumn);
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
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments()));
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
        const int maxNumBlocksByTempBytes = maxNumBlocksByInputSize;

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
                static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
                static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)*(alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize)*(alphabetSize))))
            &&
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
                static_assert(SUBMAT::numColumns == (alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)*(alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
                    static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize));
                    static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize)*(alphabetSize))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
                    static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize));
                    static_assert(SUBMAT::numColumns == (alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear) 
                || ((std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize))))

        );

        #else

        if constexpr(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
            static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
            static_assert(SUBMAT::numColumns == (alphabetSize));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
            static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize));
            static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
        }else if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
            static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
            static_assert(SUBMAT::numRows == (alphabetSize));
            static_assert(SUBMAT::numColumns == (alphabetSize));
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
    
        constexpr int paddingLetter = alphabetSize-1;
        constexpr int relaxChunkSize = 4;

        using SubstitutionScoreProviderSameQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, false> // SubjectLinearQueryLinear
        >::type;

        using SubstitutionScoreProviderDifferentQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            typename std::conditional<
                substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
                SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), false, true>,
                typename std::conditional<
                    substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, false>,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, true> //SubjectLinearQueryLinear
                >::type
            >::type
        >::type;

        using SubstitutionScoreProvider = typename std::conditional<
            InputData::isSameQueryForAll,
            SubstitutionScoreProviderSameQuery,
            SubstitutionScoreProviderDifferentQuery
        >::type;

        using ScalarScoreType = typename ScalarScoreType<ScoreType>::type;
        using MathOpsScalar = MathOps<ScalarScoreType>;
        using MathOps = MathOps<ScoreType>;
        using UpdateMaxInLastColumnOp = UpdateMaxInLastColumn<ScoreType, numItems>;
        using UpdateMaxInLastRowOp = UpdateMaxInLastRow<ScoreType>;
        using State = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>
        >::type;
        using SubjectPairLettersData = SubjectPairLettersData<alphabetSize, decltype(group)>;

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
            UpdateMaxInLastColumnOp maximumInLastColumnTracker;
            UpdateMaxInLastRowOp maximumInLastRowTracker0;
            UpdateMaxInLastRowOp maximumInLastRowTracker1;

            State state(substitutionProvider, group, maximumInLastColumnTracker, scoring);

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

                    queryLetters[i] = FuseTwoEncodedLetters{}.single(templetter0, templetter1, alphabetSize);
                }
            }
            
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

            constexpr int tileNr = 0;

            auto trackLastRow0 = [&](int row){
                if(row - group.thread_rank() == subjectLength0){
                    const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        const int position = positionOffset + i;
                        if(position < queryLength0){
                            maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                        }
                    }            
                }
            };
            auto trackLastRow1 = [&](int row){
                if(row - group.thread_rank() == subjectLength1){
                    const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        const int position = positionOffset + i;
                        if(position < queryLength1){
                            maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                        }
                    }            
                }
            };
            auto trackLastRowBoth = [&](int row){
                trackLastRow0(row);
                trackLastRow1(row);
            };

            auto doNotTrackLastRow = [](int /*row*/){};

            #if 0

            const int outputThreadRank0 = (queryLength0-1) / numItems;
            const int outputThreadRank1 = (queryLength1-1) / numItems;

            if constexpr(InputData::isSameQueryForAll){
                __builtin_assume(outputThreadRank0 == outputThreadRank1);
            }

            const int numRows0 = subjectLength0 + outputThreadRank0 + 1;
            const int numRows1 = subjectLength1 + outputThreadRank1 + 1;

            bool subject0IsShortest;
            const int numRowsForShorter = __vibmin_s32(numRows0, numRows1, &subject0IsShortest) ;
            const int numRows = max(numRows0, numRows1);

            //process until the shorter sequence is complete

            int r = processSingleTile_fromStart(
                group,
                state,
                subjectLetters,
                numRowsForShorter,
                trackLastRowBoth
            ); 

            //output alignment of shorter sequence
            {
                // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                    ScoreType temp[numItems];
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        temp[i] = maximumInLastColumnTracker.maxima[i];
                    }

                    if(subject0IsShortest){
                        const int outputRegIndex = (queryLength0-1) % numItems;
                        // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                        const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                        scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                    }else{
                        if(alignmentId1 < inputData.getNumAlignments()){
                            const int outputRegIndex = (queryLength1-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                            scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                        }
                    }
                }
            }            

            if(r < numRows){
                __builtin_assume(r > 1);

                #if 1
                
                //the compiler does not like choosing between the two sequences, so always check for both sequences if they reached the last row
                processSingleTile_continued(
                    group,
                    state,
                    subjectLetters,
                    r,
                    numRows,
                    trackLastRowBoth
                ); 

                #else

                if(subject0IsShortest){
                    //0 is not longer than 1, will process 1
                    processSingleTile_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRows,
                        trackLastRow1
                    ); 
                }else{
                    processSingleTile_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRows,
                        trackLastRow0
                    ); 
                }
                #endif
            }

            // output alignment of longer sequence
            {
                // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                    ScoreType temp[numItems];
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        temp[i] = maximumInLastColumnTracker.maxima[i];
                    }

                    if(!subject0IsShortest){
                        const int outputRegIndex = (queryLength0-1) % numItems;
                        // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                        const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                        scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                    }else{
                        if(alignmentId1 < inputData.getNumAlignments()){
                            const int outputRegIndex = (queryLength1-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                            scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                        }
                    }
                }
            }

            #else

            const int outputThreadRank0 = (queryLength0-1) / numItems;
            const int outputThreadRank1 = (queryLength1-1) / numItems;

            if constexpr(InputData::isSameQueryForAll){
                __builtin_assume(outputThreadRank0 == outputThreadRank1);
            }

            bool subject0IsShortest;
            const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
            const int longerSubjectLength = max(subjectLength0, subjectLength1);
            const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
            const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
            const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
            const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;


            int r = processSingleTile_alwayscheck_fromStart(
                group,
                state,
                subjectLetters,
                1+numRowsWithoutLastRowForAny,
                doNotTrackLastRow
            );
            r = processSingleTile_alwayscheck_continued(
                group,
                state,
                subjectLetters,
                r,
                numRowsInDPMatrixForShorter,
                trackLastRowBoth
            );
            //output alignment of shorter sequence
            {
                // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                    ScoreType temp[numItems];
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        temp[i] = maximumInLastColumnTracker.maxima[i];
                    }

                    if(subject0IsShortest){
                        const int outputRegIndex = (queryLength0-1) % numItems;
                        // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                        const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                        scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                    }else{
                        if(alignmentId1 < inputData.getNumAlignments()){
                            const int outputRegIndex = (queryLength1-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                            scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                        }
                    }
                }
            }
            r = processSingleTile_alwayscheck_continued(
                group,
                state,
                subjectLetters,
                r,
                1+numRowsWithoutLastRowForLonger,
                doNotTrackLastRow
            );
            r = processSingleTile_alwayscheck_continued(
                group,
                state,
                subjectLetters,
                r,
                numRowsInDPMatrixForLonger,
                trackLastRowBoth
            );
            // output alignment of longer sequence
            {
                // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                    ScoreType temp[numItems];
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        temp[i] = maximumInLastColumnTracker.maxima[i];
                    }

                    if(!subject0IsShortest){
                        const int outputRegIndex = (queryLength0-1) % numItems;
                        // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                        const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                        scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                    }else{
                        if(alignmentId1 < inputData.getNumAlignments()){
                            const int outputRegIndex = (queryLength1-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                            scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                        }
                    }
                }
            }

            #endif
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
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments()));
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
        const int maxNumBlocksByTempBytes = maxNumBlocksByInputSize;

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

        constexpr int expectedNumColumnsSUBMAT = alphabetSize;
        constexpr int expectedNumRowsSUBMAT = alphabetSize;

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

        constexpr int paddingLetter = alphabetSize-1;
        constexpr int relaxChunkSize = 4;

        extern __shared__ float4 externalSharedMem[];
        SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

        for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
            const int row = i / SUBMAT::numColumns;
            const int col = i % SUBMAT::numColumns;
            shared_substmat.data[row][col] = substmatPtr->data[row][col];
        }
        __syncthreads();

        using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
        using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
        using MathOps = MathOps<ScoreType>;
        using UpdateMaxInLastColumnOp = UpdateMaxInLastColumn<ScoreType, numItems>;
        using UpdateMaxInLastRowOp = UpdateMaxInLastRow<ScoreType>;
        using StateWithLastColumnMax = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>
        >::type;
        using StateWithoutLastColumnMax = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, DoNotUpdateMaxInLastColumn, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, DoNotUpdateMaxInLastColumn, relaxChunkSize>
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



        TempStorageDataType* groupTempStorage = (TempStorageDataType*)(((char*)tempStorage) + tempBytesPerGroup * groupIdInGrid);

        auto clearOutOfTileTempStorage = [&](int subjectLength){
            //this is only required for local alignment to avoid modifiying the stored maximum by oob computations

            // if(group.thread_rank() < group.size() - 1){
            //     groupTempStorage[subjectLength + group.thread_rank()] = ScoreType{};
            // }
        };

        for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){

            const auto* query = inputData.getQuery(alignmentId);
            const int queryLength = inputData.getQueryLength(alignmentId);
            const int numTiles = SDIV(queryLength, groupsize * numItems);

            if(numTiles == 1){
                constexpr int tileNr = 0;

                UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                UpdateMaxInLastRowOp maximumInLastRowTracker;

                int queryLetters[numItems];
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                auto loadQueryLetters = [&](int tileNr){
                    #pragma unroll
                    for (int i=0; i < numItems; i++) {
                        const int index = tileNr * groupsize * numItems + numItems * group.thread_rank()+i;
                        if (index >= queryLength) queryLetters[i] = paddingLetter;
                        else queryLetters[i] = query[index]; 
                    }
                };
                loadQueryLetters(tileNr);

                StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);

                const int subjectLength = inputData.getSubjectLength(alignmentId);
                const std::int8_t* const subjectData = inputData.getSubject(alignmentId);
                SubjectLettersData subjectLetters(group, subjectData, subjectLength);        
        
                subjectLetters.loadNext4Letters();
                state.initScores(0, FirstLeftBorder<ScoreType>{});

                auto trackLastRow = [&](int row){
                    if(row - group.thread_rank() == subjectLength){
                        const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            const int position = positionOffset + i;
                            if(position < queryLength){
                                maximumInLastRowTracker(state.scoresM[i], tileNr, i);
                            }
                        }            
                    }
                };
        
                const int outputThreadRank = (queryLength-1) / numItems;
                const int numRows = subjectLength + outputThreadRank + 1;

                processSingleTile_fromStart(
                    group,
                    state,
                    subjectLetters,
                    numRows,
                    trackLastRow
                );    
                
                const ScoreType groupmaxLastRow = MathOps::reduce_max(group, maximumInLastRowTracker.maximum);
                const int outputRegIndex = (queryLength-1) % numItems;
                if(group.thread_rank() == outputThreadRank){
                    ScoreType temp[numItems];
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        temp[i] = maximumInLastColumnTracker.maxima[i];
                    }
                    const ScoreType maxLastColumn = temp[outputRegIndex];
                    scoreOutput[alignmentId] = MathOps::max(groupmaxLastRow, maxLastColumn);
                }

            }else{
                UpdateMaxInLastRowOp maximumInLastRowTracker;
                LeftBorderType leftBorder;

                int subjectLength = 0;
                const std::int8_t* subjectData = nullptr;
                int queryLetters[numItems];
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                
                auto loadQueryLetters = [&](int tileNr, int (&queryLetters)[numItems]){
                    #pragma unroll
                    for (int i=0; i < numItems; i++) {
                        const int index = tileNr * groupsize * numItems + numItems * group.thread_rank()+i;
                        if (index >= queryLength) queryLetters[i] = paddingLetter;
                        else queryLetters[i] = query[index]; 
                    }
                };

                /* 
                    -----------------------
                    Process tile 0
                    ----------------------- 
                */
                {
                    constexpr int tileNr = 0;

                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    DoNotUpdateMaxInLastColumn doNotUpdateLastColumnMax;
                    StateWithoutLastColumnMax state(substitutionProvider, group, doNotUpdateLastColumnMax, scoring);

                    subjectLength = inputData.getSubjectLength(alignmentId);
                    subjectData = inputData.getSubject(alignmentId);
                    clearOutOfTileTempStorage(subjectLength);

                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);
                    subjectLetters.loadNext4Letters();

                    LastColumnInLastThread lastColumn;

                    state.initScores(0, FirstLeftBorder<ScoreType>{});


                    auto trackLastRow = [&](int row){
                        if(row - group.thread_rank() == subjectLength){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength){
                                    maximumInLastRowTracker(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };

                    const int numRows = subjectLength + (group.size()-1) + 1;

                    TempHandler tempHandler(group, groupTempStorage);
                    processFirstTile_fromStart(
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRow,
                        subjectLength,
                        lastColumn,
                        tempHandler
                    );

                    //flush any unsaved variables to temp storage
                    const int numComputedNonOOBRowsInTile = subjectLength;
                    tempStorageTileCompleted(group, numComputedNonOOBRowsInTile, lastColumn, tempHandler);

                }

                //process intermediate tiles
                for(int tileNr = 1; tileNr < numTiles-1; tileNr++){
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    DoNotUpdateMaxInLastColumn doNotUpdateLastColumnMax;
                    StateWithoutLastColumnMax state(substitutionProvider, group, doNotUpdateLastColumnMax, scoring);
                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);

                    auto trackLastRow = [&](int row){
                        if(row - group.thread_rank() == subjectLength){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength){
                                    maximumInLastRowTracker(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };

                    const int numRows = subjectLength + (group.size()-1) + 1;

                    processIntermediateTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRow,
                        subjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    //flush any unsaved variables to temp storage
                    const int numComputedNonOOBRowsInTile = subjectLength;
                    tempStorageTileCompleted(group, numComputedNonOOBRowsInTile, lastColumn, tempHandler);
                }

                //process last tile
                {
                    const int tileNr = numTiles-1;
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                    StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);
                    SubjectLettersData subjectLetters(group, subjectData, subjectLength);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    state.initScores(tileNr, leftBorder);

                    auto trackLastRow = [&](int row){
                        if(row - group.thread_rank() == subjectLength){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength){
                                    maximumInLastRowTracker(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };

                    const int queryLengthInLastTile = queryLength - (numTiles-1) * (groupsize * numItems);
                    const int outputThreadRank = (queryLengthInLastTile-1) / numItems;
                    const int numRows = subjectLength + outputThreadRank + 1;

                    processLastTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRow,
                        subjectLength,
                        leftBorder,
                        tempHandler
                    );

                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, maximumInLastRowTracker.maximum);            
                    const int outputRegIndex = (queryLengthInLastTile-1) % numItems;
                    if(group.thread_rank() == outputThreadRank){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                        }
                        const ScoreType maxLastColumn = temp[outputRegIndex];
                        // printf("groupmaxLastRow %f, maxLastColumn %f\n", groupmaxLastRow, maxLastColumn);
                        scoreOutput[alignmentId] = MathOps::max(groupmaxLastRow, maxLastColumn);
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
        int maximumSequenceLength,
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
            CUDACHECK(cudaMemsetAsync(d_scoreOutput, 0, sizeof(int) * inputData.getNumAlignments()));
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
        const int maxSubjectLengthPadded = maximumSequenceLength + groupsize;
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
                static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
                static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)*(alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize)*(alphabetSize))))
            &&
            /*
            if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear){
                static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                static_assert(SUBMAT::numRows == (alphabetSize)*(alphabetSize));
                static_assert(SUBMAT::numColumns == (alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)*(alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared){
                    static_assert(std::is_same_v<ScoreType, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize));
                    static_assert(SUBMAT::numColumns == (alphabetSize)*(alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared) 
                || ((std::is_same_v<ScoreType, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize)*(alphabetSize))))
            &&
            /*
                if (substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear){
                    static_assert(std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>);
                    static_assert(SUBMAT::numRows == (alphabetSize));
                    static_assert(SUBMAT::numColumns == (alphabetSize));
            */
            (!(substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQueryLinear) 
                || ((std::is_same_v<typename ScalarScoreType<ScoreType>::type, typename SUBMAT::value_type>) 
                    && (SUBMAT::numRows == (alphabetSize)) 
                    && (SUBMAT::numColumns == (alphabetSize))))

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
        constexpr int tileSize = groupsize * numItems;

        constexpr int paddingLetter = alphabetSize-1;
        constexpr int relaxChunkSize = 4;

        using SubstitutionScoreProviderSameQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, false> // SubjectLinearQueryLinear
        >::type;

        using SubstitutionScoreProviderDifferentQuery = typename std::conditional<
            substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQuerySquared,
            SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>,
            typename std::conditional<
                substmatDimMode == SubstitutionMatrixDimensionMode::SubjectSquaredQueryLinear,
                SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), false, true>,
                typename std::conditional<
                    substmatDimMode == SubstitutionMatrixDimensionMode::SubjectLinearQuerySquared,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, false>,
                    SubstitutionMatrixPackingSubstitutionScoreProvider<SUBMAT, ScoreType, numItems, (alphabetSize), true, true> //SubjectLinearQueryLinear
                >::type
            >::type
        >::type;

        using SubstitutionScoreProvider = typename std::conditional<
            InputData::isSameQueryForAll,
            SubstitutionScoreProviderSameQuery,
            SubstitutionScoreProviderDifferentQuery
        >::type;

        using ScalarScoreType = typename ScalarScoreType<ScoreType>::type;
        using MathOpsScalar = MathOps<ScalarScoreType>;
        using MathOps = MathOps<ScoreType>;
        using UpdateMaxInLastColumnOp = UpdateMaxInLastColumn<ScoreType, numItems>;
        using UpdateMaxInLastRowOp = UpdateMaxInLastRow<ScoreType>;
        using StateWithLastColumnMax = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>
        >::type;
        using StateWithoutLastColumnMax = typename std::conditional<
            penaltyType == PenaltyType::Linear,
            SemiglobalAlignmentLinearGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, DoNotUpdateMaxInLastColumn, relaxChunkSize>,
            SemiglobalAlignmentAffineGapState_half2OrShort2<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, DoNotUpdateMaxInLastColumn, relaxChunkSize>
        >::type;
        using SubjectPairLettersData = SubjectPairLettersData<alphabetSize, decltype(group)>;
        
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

        TempStorageDataType* groupTempStorage = (TempStorageDataType*)(((char*)tempStorage) + tempBytesPerGroup * groupIdInGrid);
        // if(threadIdx.x == 0) printf("tempBytesPerGroup %lu\n", tempBytesPerGroup);
        auto clearOutOfTileTempStorage = [&](int subjectLength){
            // if(group.thread_rank() < group.size() - 1){
            //     groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
            // }
        };

        for(int alignmentId = 2*groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid*2){
            UpdateMaxInLastRowOp maximumInLastRowTracker0;
            UpdateMaxInLastRowOp maximumInLastRowTracker1;

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

                        // printf("t %d, i %d, loaded(%d %d)\n", group.thread_rank(), i, templetter0, templetter1);
                        queryLetters[i] = FuseTwoEncodedLetters{}.single(templetter0, templetter1, alphabetSize);
                    }
                }
                // for(int t = 0; t < group.size(); t++){
                //     if(t == group.thread_rank()){
                //         for (int i=0; i < numItems; i++) {
                //             printf("%d ", queryLetters[i]);
                //         }
                //         printf("\n");
                //     }
                // }
                
            };

            auto doNotTrackLastRow = [](int /*row*/){};

            const int numTiles0 = SDIV(queryLength0, tileSize);
            const int numTiles1 = SDIV(queryLength1, tileSize);
            // if(group.thread_rank() == 0){
            //     printf("numTiles0 %d, numTiles1 %d\n", numTiles0, numTiles1);
            // }
        
            if(numTiles0 == 1 && numTiles1 == 1){
                constexpr int tileNr = 0;
                int queryLetters[numItems];
                loadQueryLetters(tileNr, queryLetters);

                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);
                auto trackLastRow0 = [&](int row){
                    if(row - group.thread_rank() == subjectLength0){
                        const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            const int position = positionOffset + i;
                            if(position < queryLength0){
                                maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                            }
                        }            
                    }
                };
                auto trackLastRow1 = [&](int row){
                    if(row - group.thread_rank() == subjectLength1){
                        const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            const int position = positionOffset + i;
                            if(position < queryLength1){
                                maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                            }
                        }            
                    }
                };
                auto trackLastRowBoth = [&](int row){
                    trackLastRow0(row);
                    trackLastRow1(row);
                };

                SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);
        
                subjectLetters.loadNext4Letters();
                state.initScores(0, FirstLeftBorder<ScoreType>{});

                const int queryLengthInLastTile0 = queryLength0 - (numTiles0-1) * tileSize;
                const int queryLengthInLastTile1 = queryLength1 - (numTiles1-1) * tileSize;
                const int outputThreadRank0 = (queryLengthInLastTile0-1) / numItems;
                const int outputThreadRank1 = (queryLengthInLastTile1-1) / numItems;

                if constexpr(InputData::isSameQueryForAll){
                    __builtin_assume(outputThreadRank0 == outputThreadRank1);
                }

                #if 1
                const int numRows0 = subjectLength0 + outputThreadRank0 + 1;
                const int numRows1 = subjectLength1 + outputThreadRank1 + 1;

                bool subject0IsShortest;
                const int numRowsForShorter = __vibmin_s32(numRows0, numRows1, &subject0IsShortest) ;
                const int numRows = max(numRows0, numRows1);

                int r = processSingleTile_fromStart(
                    group,
                    state,
                    subjectLetters,
                    numRowsForShorter,
                    trackLastRowBoth
                ); 

                //output alignment of shorter sequence
                {
                    // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                    if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                        }

                        if(subject0IsShortest){
                            const int outputRegIndex = (queryLength0-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                            scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                        }else{
                            if(alignmentId1 < inputData.getNumAlignments()){
                                const int outputRegIndex = (queryLength1-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                            }
                        }
                    }
                } 

                //process remaining rows for the longer sequence

                if(r < numRows){
                    __builtin_assume(r > 1);

                    processSingleTile_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRows,
                        trackLastRowBoth
                    ); 

                }

                // output alignment of longer sequence
                {
                    // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                    if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                        }

                        if(!subject0IsShortest){
                            const int outputRegIndex = (queryLength0-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                            scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                        }else{
                            if(alignmentId1 < inputData.getNumAlignments()){
                                const int outputRegIndex = (queryLength1-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                            }
                        }
                    }
                }

                #else
                bool subject0IsShortest;
                const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
                const int longerSubjectLength = max(subjectLength0, subjectLength1);
                const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
                const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
                const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
                const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;
    
                int r = processSingleTile_alwayscheck_fromStart(
                    group,
                    state,
                    subjectLetters,
                    1+numRowsWithoutLastRowForAny,
                    doNotTrackLastRow
                );
                r = processSingleTile_alwayscheck_continued(
                    group,
                    state,
                    subjectLetters,
                    r,
                    numRowsInDPMatrixForShorter,
                    trackLastRowBoth
                );
                //output alignment of shorter sequence
                {
                    // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                    if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                        }
    
                        if(subject0IsShortest){
                            const int outputRegIndex = (queryLength0-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                            scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                        }else{
                            if(alignmentId1 < inputData.getNumAlignments()){
                                const int outputRegIndex = (queryLength1-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                            }
                        }
                    }
                }
                r = processSingleTile_alwayscheck_continued(
                    group,
                    state,
                    subjectLetters,
                    r,
                    1+numRowsWithoutLastRowForLonger,
                    doNotTrackLastRow
                );
                r = processSingleTile_alwayscheck_continued(
                    group,
                    state,
                    subjectLetters,
                    r,
                    numRowsInDPMatrixForLonger,
                    trackLastRowBoth
                );
                // output alignment of longer sequence
                {
                    // const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum, maximumInLastRowTracker1.maximum));
                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                    if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                        }
    
                        if(!subject0IsShortest){
                            const int outputRegIndex = (queryLength0-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                            scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                        }else{
                            if(alignmentId1 < inputData.getNumAlignments()){
                                const int outputRegIndex = (queryLength1-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                            }
                        }
                    }
                }

                #endif

            }else{
                LeftBorderType leftBorder;

                const int longerSubjectLength = max(subjectLength0, subjectLength1);
                const int numTilesNotLastForAny = min(numTiles0-1, numTiles1-1);

                // if(group.thread_rank() == 0){
                //     printf("longerSubjectLength %d, numTilesNotLastForAny %d\n", longerSubjectLength, numTilesNotLastForAny);
                // }

                if(numTilesNotLastForAny == 0){
                    /* 
                        -----------------------
                        Process tile 0, 1 subject/query will finish
                        ----------------------- 
                    */
                    constexpr int tileNr = 0;
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, tile %d\n", __LINE__, tileNr);
                    // }

                    clearOutOfTileTempStorage(longerSubjectLength);
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                    StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);
                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            // printf("row %d, tid %d, update last row max 0\n", row, group.thread_rank());
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            // printf("row %d, tid %d, update last row max 1\n", row, group.thread_rank());
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();
                    state.initScores(0, FirstLeftBorder<ScoreType>{});

                    LastColumnInLastThread lastColumn;
                    TempHandler tempHandler(group, groupTempStorage);
                    const int queryLength0InTile = min(queryLength0, tileSize);
                    const int queryLength1InTile = min(queryLength1, tileSize);
                    const int outputThreadRank0 = (queryLength0InTile-1) / numItems;
                    const int outputThreadRank1 = (queryLength1InTile-1) / numItems;
        
                    const int numRows0 = subjectLength0 + outputThreadRank0 + 1;
                    const int numRows1 = subjectLength1 + outputThreadRank1 + 1;
                    #if 1

                    // if(group.thread_rank() == 0){
                    //     printf("line %d, queryLength0InTile %d, queryLength1InTile %d, numRows0 %d, numRows1 %d\n",
                    //         __LINE__, queryLength0InTile, queryLength1InTile, numRows0 , numRows1
                    //     );
                    // }
    
                    bool subject0IsShortest;
                    const int numRowsForShorter = __vibmin_s32(numRows0, numRows1, &subject0IsShortest) ;
                    const int numRows = max(numRows0, numRows1);

                    
                    int r = processFirstTile_fromStart(
                        group,
                        state,
                        subjectLetters,
                        numRowsForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, after shorter, r = %d\n", __LINE__, r);
                    // }
                    // state.printState();
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, output shorter\n", __LINE__);
                    // }
                    // state.printState();
                    const int numTilesOfShorter = subject0IsShortest ? numTiles0 : numTiles1;
                    if(numTilesOfShorter == 1){
                        //output alignment of shorter sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        // printf("line %d, groupmaxLastRow (%d %d)\n", __LINE__, int(groupmaxLastRow.x), int(groupmaxLastRow.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                                // printf("(%d %d) ", int(temp[i].x), int(temp[i].y));
                            }
                            // printf("\n");

                            if(subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    //process remaining rows for the longer sequence
                    if(r < numRows){
                        __builtin_assume(r > 1);

                        processFirstTile_continued(
                            group,
                            state,
                            subjectLetters,
                            r,
                            numRows,
                            trackLastRowBoth,
                            longerSubjectLength,
                            lastColumn,
                            tempHandler
                        );

                        // if(group.thread_rank() == 0){
                        //     printf("line %d, after continue\n", __LINE__);
                        // }
                        // state.printState();
                    }

                    const int numTilesOfLonger = subject0IsShortest ? numTiles1 : numTiles0;
                    if(numTilesOfLonger == 1){
                        //output alignment of longer sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        // printf("line %d, groupmaxLastRow (%d %d)\n", __LINE__, int(groupmaxLastRow.x), int(groupmaxLastRow.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                                // printf("(%d %d) ", int(temp[i].x), int(temp[i].y));
                            }
                            // printf("\n");

                            if(!subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    #else

                    bool subject0IsShortest;
                    const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
                    const int longerSubjectLength = max(subjectLength0, subjectLength1);
                    const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
                    const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
                    const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
                    const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;
        

                    int r = processFirstTile_alwayscheck_fromStart(
                        group,
                        state,
                        subjectLetters,
                        1+numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );

                    const int numTilesOfShorter = subject0IsShortest ? numTiles0 : numTiles1;
                    if(numTilesOfShorter == 1){
                        //output alignment of shorter sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        // printf("line %d, groupmaxLastRow (%d %d)\n", __LINE__, int(groupmaxLastRow.x), int(groupmaxLastRow.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                                // printf("(%d %d) ", int(temp[i].x), int(temp[i].y));
                            }
                            // printf("\n");

                            if(subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        1+numRowsWithoutLastRowForLonger,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForLonger,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );

                    const int numTilesOfLonger = subject0IsShortest ? numTiles1 : numTiles0;
                    if(numTilesOfLonger == 1){
                        //output alignment of longer sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        // printf("line %d, groupmaxLastRow (%d %d)\n", __LINE__, int(groupmaxLastRow.x), int(groupmaxLastRow.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                                // printf("(%d %d) ", int(temp[i].x), int(temp[i].y));
                            }
                            // printf("\n");

                            if(!subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }


                    #endif

                    //flush any unsaved variables to temp storage
                    //tricky bit: the subject which was continued does not need to be the longer subject, if the longer subject terminates in the current tile
                    const int numComputedNonOOBRowsInTile_maybefinished = numTiles0 == 1 ? subjectLength1 : subjectLength0;
                    const int numComputedRowsInTile_maybefinished = -1 + ((numTiles0 == 1) ? numRows1 : numRows0);
                    const int numComputedRowsInTile_finished = -1 + ((numTiles0 == 1) ? numRows0 : numRows1);
                    tempStorageTileCompleted_half2OrShort2_oneAlignmentFinished(
                        group,
                        numComputedNonOOBRowsInTile_maybefinished,
                        numComputedRowsInTile_maybefinished,
                        numComputedRowsInTile_finished,
                        lastColumn,
                        tempHandler
                    );

                    // group.sync();
                    // if(group.thread_rank() == 0){
                    //     printf("temp storage:\n");
                    //     for(int i = 0; i < longerSubjectLength; i++){
                    //         if constexpr(penaltyType == PenaltyType::Linear){
                    //             printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                    //         }
                    //     }
                    //     printf("\n");
                    // }
                }else{
                    //first tile is full for each query
                    /* 
                        -----------------------
                        Process tile 0
                        ----------------------- 
                    */
                    constexpr int tileNr = 0;
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, tile %d\n", __LINE__, tileNr);
                    // }

                    clearOutOfTileTempStorage(longerSubjectLength);
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                    DoNotUpdateMaxInLastColumn doNotUpdateLastColumnMax;
                    StateWithoutLastColumnMax state(substitutionProvider, group, doNotUpdateLastColumnMax, scoring);

                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    
                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();
                    state.initScores(0, FirstLeftBorder<ScoreType>{});

                    LastColumnInLastThread lastColumn;
                    TempHandler tempHandler(group, groupTempStorage);
                    #if 1

                    const int numRows = longerSubjectLength + (group.size()-1) + 1;
                    
                    processFirstTile_fromStart(
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );

                    #else

                    const int outputThreadRank0 = group.size()-1;
                    const int outputThreadRank1 = group.size()-1;

                    bool subject0IsShortest;
                    const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
                    const int longerSubjectLength = max(subjectLength0, subjectLength1);
                    const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
                    const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
                    const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
                    const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;

                    int r = processFirstTile_alwayscheck_fromStart(
                        group,
                        state,
                        subjectLetters,
                        1+numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        1+numRowsWithoutLastRowForLonger,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );
                    r = processFirstTile_alwayscheck_continued(
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForLonger,
                        trackLastRowBoth,
                        longerSubjectLength,
                        lastColumn,
                        tempHandler
                    );

                    #endif
                    //flush any unsaved variables to temp storage
                    const int numComputedNonOOBRowsInTile = longerSubjectLength;
                    tempStorageTileCompleted(group, numComputedNonOOBRowsInTile, lastColumn, tempHandler);
                    // state.printState();
                    // group.sync();
                    // if(group.thread_rank() == 0){
                    //     printf("temp storage:\n");
                    //     for(int i = 0; i < longerSubjectLength; i++){
                    //         if constexpr(penaltyType == PenaltyType::Linear){
                    //             printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                    //         }
                    //     }
                    //     printf("\n");
                    // }
                }

                //process intermediate tiles. it is not the last tile for either query
                for(int tileNr = 1; tileNr < numTilesNotLastForAny; tileNr++){
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, tile %d\n", __LINE__, tileNr);
                    // }

                    DoNotUpdateMaxInLastColumn doNotUpdateLastColumnMax;
                    StateWithoutLastColumnMax state(substitutionProvider, group, doNotUpdateLastColumnMax, scoring);

                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);

                    #if 1
                    const int numRows = longerSubjectLength + (group.size()-1) + 1;

                    processIntermediateTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    #else

                    const int outputThreadRank0 = group.size()-1;
                    const int outputThreadRank1 = group.size()-1;

                    bool subject0IsShortest;
                    const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
                    const int longerSubjectLength = max(subjectLength0, subjectLength1);
                    const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
                    const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
                    const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
                    const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;

                    int r = processIntermediateTile_alwayscheck_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        1+numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        1+numRowsWithoutLastRowForLonger,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForLonger,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    #endif

                    //flush any unsaved variables to temp storage
                    const int numComputedNonOOBRowsInTile = longerSubjectLength;
                    tempStorageTileCompleted(group, numComputedNonOOBRowsInTile, lastColumn, tempHandler);

                    // group.sync();
                    // if(group.thread_rank() == 0){
                    //     printf("temp storage:\n");
                    //     for(int i = 0; i < longerSubjectLength; i++){
                    //         if constexpr(penaltyType == PenaltyType::Linear){
                    //             printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                    //         }
                    //     }
                    //     printf("\n");
                    // }
                }
                //process tile which is last for at least one query. case numTilesNotLastForAny == 0 is handled above
                if(numTilesNotLastForAny > 0){
                    const int tileNr = numTilesNotLastForAny;
                    if(group.thread_rank() == 0){
                        // printf("line %d, tile %d\n", __LINE__, tileNr);
                        // printf("temp storage:\n");
                        // for(int i = 0; i < longerSubjectLength; i++){
                        //     if constexpr(penaltyType == PenaltyType::Linear){
                        //         printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                        //     }
                        // }
                        // printf("\n");
                    }

                    // clearOutOfTileTempStorage(longerSubjectLength);
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                    StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);

                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);

                    // state.printState();

                    const int queryLength0InTile = min(queryLength0 - (tileNr) * tileSize, tileSize);
                    const int queryLength1InTile = min(queryLength1 - (tileNr) * tileSize, tileSize);
                    const int outputThreadRank0 = (queryLength0InTile-1) / numItems;
                    const int outputThreadRank1 = (queryLength1InTile-1) / numItems;
                    if constexpr(InputData::isSameQueryForAll){
                        __builtin_assume(outputThreadRank0 == outputThreadRank1);
                    }
                    const int numRows0 = subjectLength0 + outputThreadRank0 + 1;
                    const int numRows1 = subjectLength1 + outputThreadRank1 + 1;
        
                    #if 1

                    // if(group.thread_rank() == 0){
                    //     printf("tileNr %d, queryLength0InTile %d, queryLength1InTile %d, outputThreadRank0 %d, outputThreadRank1 %d"
                    //         ", numRows0 %d, numRows1 %d\n", 
                    //         tileNr, queryLength0InTile, queryLength1InTile, outputThreadRank0, outputThreadRank1,
                    //         numRows0, numRows1
                    //     );
                    // }
    
                    bool subject0IsShortest;
                    const int numRowsForShorter = __vibmin_s32(numRows0, numRows1, &subject0IsShortest) ;
                    const int numRows = max(numRows0, numRows1);
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, numRowsForShorter %d, numRows %d\n", 
                    //         __LINE__, numRowsForShorter, numRows
                    //     );
                    // }

                    int r = processIntermediateTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRowsForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    // state.printState();

                    const int numTilesOfShorter = subject0IsShortest ? numTiles0 : numTiles1;
                    if(numTilesOfShorter == tileNr+1){
                        // if(group.thread_rank() == 0){
                        //     printf("line %d, state before output short\n", 
                        //         __LINE__
                        //     );
                        // }
                        // group.sync();
                        // state.printState();
                        // group.sync();
                        //output alignment of shorter sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                            }

                            if(subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    if(r < numRows){
                        __builtin_assume(r > 1);

                        // if(group.thread_rank() == 0){
                        //     printf("line %d, continue. r %d, numRows %d\n", 
                        //         __LINE__, r, numRows
                        //     );
                        // }
                        processIntermediateTile_continued(
                            tileNr,
                            group,
                            state,
                            subjectLetters,
                            r,
                            numRows,
                            trackLastRowBoth,
                            longerSubjectLength,
                            leftBorder,
                            lastColumn,
                            tempHandler
                        );
                    }

                    // state.printState();

                    const int numTilesOfLonger = subject0IsShortest ? numTiles1 : numTiles0;
                    if(numTilesOfLonger == tileNr+1){
                        // if(group.thread_rank() == 0){
                        //     printf("line %d, state before output long\n", 
                        //         __LINE__
                        //     );
                        // }
                        // group.sync();
                        // state.printState();
                        // group.sync();

                        //output alignment of longer sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                            }

                            if(!subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    #else

                    bool subject0IsShortest;
                    const int shorterSubjectLength = __vibmin_s32(subjectLength0, subjectLength1, &subject0IsShortest);
                    const int longerSubjectLength = max(subjectLength0, subjectLength1);
                    const int numRowsWithoutLastRowForAny = shorterSubjectLength-1;
                    const int numRowsInDPMatrixForShorter = shorterSubjectLength + (subject0IsShortest ? outputThreadRank0 : outputThreadRank1) + 1;
                    const int numRowsWithoutLastRowForLonger = longerSubjectLength-1;
                    const int numRowsInDPMatrixForLonger = longerSubjectLength +  (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)  + 1;

                    int r = processIntermediateTile_alwayscheck_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        1+numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForShorter,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    const int numTilesOfShorter = subject0IsShortest ? numTiles0 : numTiles1;
                    if(numTilesOfShorter == tileNr+1){
                        // if(group.thread_rank() == 0){
                        //     printf("line %d, state before output short\n", 
                        //         __LINE__
                        //     );
                        // }
                        // group.sync();
                        // state.printState();
                        // group.sync();
                        //output alignment of shorter sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank0 : outputThreadRank1)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                            }

                            if(subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }

                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        1+numRowsWithoutLastRowForLonger,
                        doNotTrackLastRow,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForLonger,
                        trackLastRowBoth,
                        longerSubjectLength,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    const int numTilesOfLonger = subject0IsShortest ? numTiles1 : numTiles0;
                    if(numTilesOfLonger == tileNr+1){
                        // if(group.thread_rank() == 0){
                        //     printf("line %d, state before output long\n", 
                        //         __LINE__
                        //     );
                        // }
                        // group.sync();
                        // state.printState();
                        // group.sync();

                        //output alignment of longer sequence
                        const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                        if(group.thread_rank() == (subject0IsShortest ? outputThreadRank1 : outputThreadRank0)){
                            ScoreType temp[numItems];
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                temp[i] = maximumInLastColumnTracker.maxima[i];
                            }

                            if(!subject0IsShortest){
                                const int outputRegIndex = (queryLength0InTile-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                                scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                            }else{
                                if(alignmentId1 < inputData.getNumAlignments()){
                                    const int outputRegIndex = (queryLength1InTile-1) % numItems;
                                    // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                    const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                    scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                                }
                            }
                        }
                    }


                    #endif

                    //flush any unsaved variables to temp storage
                    //tricky bit: the subject which was continued does not need to be the longer subject, if the longer subject terminates in the current tile
                    const int numComputedNonOOBRowsInTile_maybefinished = (numTiles0 == tileNr+1) ? subjectLength1 : subjectLength0;
                    const int numComputedRowsInTile_maybefinished = -1 + ((numTiles0 == tileNr+1) ? numRows1 : numRows0);
                    const int numComputedRowsInTile_finished = -1 + ((numTiles0 == tileNr+1) ? numRows0 : numRows1);
                    tempStorageTileCompleted_half2OrShort2_oneAlignmentFinished(
                        group,
                        numComputedNonOOBRowsInTile_maybefinished,
                        numComputedRowsInTile_maybefinished,
                        numComputedRowsInTile_finished,
                        lastColumn,
                        tempHandler
                    );

                    // group.sync();
                    // if(group.thread_rank() == 0){
                    //     printf("temp storage:\n");
                    //     for(int i = 0; i < longerSubjectLength; i++){
                    //         if constexpr(penaltyType == PenaltyType::Linear){
                    //             printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                    //         }
                    //     }
                    //     printf("\n");
                    // }
                }

                const int maxTiles = max(numTiles0, numTiles1);
                const int subjectLengthOfUnfinishedAlignment = maxTiles == numTiles0 ? subjectLength0 : subjectLength1;
                const int queryLengthOfUnfinishedAlignment = maxTiles == numTiles0 ? queryLength0 : queryLength1;

                // if(group.thread_rank() == 0){
                //     printf("numTilesNotLastForAny %d, maxTiles %d, subjectLengthOfUnfinishedAlignment %d"
                //         ", queryLengthOfUnfinishedAlignment %d\n",
                //         numTilesNotLastForAny, maxTiles, 
                //         subjectLengthOfUnfinishedAlignment,
                //         queryLengthOfUnfinishedAlignment
                //     );
                // }

                //process the remaining tiles. only 1 alignment is active
                for(int tileNr = numTilesNotLastForAny+1; tileNr < maxTiles-1; tileNr++){
                    if(group.thread_rank() == 0){
                        // printf("line %d, tile %d\n", __LINE__, tileNr);
                        // printf("temp storage:\n");
                        // for(int i = 0; i < longerSubjectLength; i++){
                        //     if constexpr(penaltyType == PenaltyType::Linear){
                        //         printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                        //     }
                        // }
                        // printf("\n");
                    }
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    DoNotUpdateMaxInLastColumn doNotUpdateLastColumnMax;
                    StateWithoutLastColumnMax state(substitutionProvider, group, doNotUpdateLastColumnMax, scoring);

                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    LastColumnInLastThread lastColumn;
                    state.initScores(tileNr, leftBorder);

                    #if 1
                    const int numRows = subjectLengthOfUnfinishedAlignment + (group.size()-1) + 1;
                    // if(group.thread_rank() == 0){
                    //     printf("line %d, numRows %d\n", __LINE__, numRows);
                    // }
                    processIntermediateTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRowBoth,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    #else

                    const int numRowsWithoutLastRowForAny = subjectLengthOfUnfinishedAlignment-1;
                    const int numRowsInDPMatrixForUnfinished = subjectLengthOfUnfinishedAlignment + (group.size()-1) + 1;

                    int r = processIntermediateTile_alwayscheck_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );
                    r = processIntermediateTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForUnfinished,
                        trackLastRowBoth,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        lastColumn,
                        tempHandler
                    );

                    #endif

                    //flush any unsaved variables to temp storage
                    const int numComputedNonOOBRowsInTile = subjectLengthOfUnfinishedAlignment;
                    tempStorageTileCompleted(group, numComputedNonOOBRowsInTile, lastColumn, tempHandler);

                    // group.sync();
                    // if(group.thread_rank() == 0){
                    //     printf("temp storage:\n");
                    //     for(int i = 0; i < longerSubjectLength; i++){
                    //         if constexpr(penaltyType == PenaltyType::Linear){
                    //             printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                    //         }
                    //     }
                    //     printf("\n");
                    // }
                }

                //process last tile
                if(numTilesNotLastForAny+1 < maxTiles){
                    const int tileNr = maxTiles-1;
                    if(group.thread_rank() == 0){
                        // printf("line %d, tile %d\n", __LINE__, tileNr);
                        // printf("temp storage:\n");
                        // for(int i = 0; i < longerSubjectLength; i++){
                        //     if constexpr(penaltyType == PenaltyType::Linear){
                        //         printf("(%d %d)", int(groupTempStorage[i].x), int(groupTempStorage[i].y));
                        //     }
                        // }
                        // printf("\n");
                    }
                    int queryLetters[numItems];
                    loadQueryLetters(tileNr, queryLetters);

                    SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                    UpdateMaxInLastColumnOp maximumInLastColumnTracker;
                    StateWithLastColumnMax state(substitutionProvider, group, maximumInLastColumnTracker, scoring);

                    auto trackLastRow0 = [&](int row){
                        if(row - group.thread_rank() == subjectLength0){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength0){
                                    maximumInLastRowTracker0(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRow1 = [&](int row){
                        if(row - group.thread_rank() == subjectLength1){
                            const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                            #pragma unroll
                            for(int i = 0; i < numItems; i++){
                                const int position = positionOffset + i;
                                if(position < queryLength1){
                                    maximumInLastRowTracker1(state.scoresM[i], tileNr, i);
                                }
                            }            
                        }
                    };
                    auto trackLastRowBoth = [&](int row){
                        trackLastRow0(row);
                        trackLastRow1(row);
                    };

                    SubjectPairLettersData subjectLetters(group, subjectData0, subjectLength0,  subjectData1, subjectLength1);

                    subjectLetters.loadNext4Letters();

                    TempHandler tempHandler(group, groupTempStorage);
                    leftBorder.setPayload(tempHandler.load());

                    state.initScores(tileNr, leftBorder);

                    const int queryLengthInLastTile = queryLengthOfUnfinishedAlignment - (tileNr) * (groupsize * numItems);
                    const int lastColThreadRank = (queryLengthInLastTile-1) / numItems;

                    #if 1
                    const int numRows = subjectLengthOfUnfinishedAlignment + lastColThreadRank + 1;

                    // if(group.thread_rank() == 0){
                    //     printf("line %d, tile %d, queryLengthInLastTile %d, lastColThreadRank %d, numRows %d\n", 
                    //         __LINE__, tileNr, queryLengthInLastTile, lastColThreadRank, numRows);

                    // }

                    processLastTile_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRows,
                        trackLastRowBoth,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        tempHandler
                    );

                    #else

                    const int numRowsWithoutLastRowForAny = subjectLengthOfUnfinishedAlignment-1;
                    const int numRowsInDPMatrixForUnfinished = subjectLengthOfUnfinishedAlignment + lastColThreadRank + 1;

                    int r = processLastTile_alwayscheck_fromStart(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        numRowsWithoutLastRowForAny,
                        doNotTrackLastRow,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        tempHandler
                    );
                    r = processLastTile_alwayscheck_continued(
                        tileNr,
                        group,
                        state,
                        subjectLetters,
                        r,
                        numRowsInDPMatrixForUnfinished,
                        trackLastRowBoth,
                        subjectLengthOfUnfinishedAlignment,
                        leftBorder,
                        tempHandler
                    );


                    #endif

                    const int queryLengthInLastTile0 = queryLength0 - (numTiles0-1) * tileSize;
                    const int queryLengthInLastTile1 = queryLength1 - (numTiles1-1) * tileSize;
                    const int outputThreadRank0 = (queryLengthInLastTile0-1) / numItems;
                    const int outputThreadRank1 = (queryLengthInLastTile1-1) / numItems;

                    // if(group.thread_rank() == 0){
                    //     printf("line %d, output final. queryLengthInLastTile0 %d, queryLengthInLastTile1 %d, outputThreadRank0 %d, outputThreadRank1 %d\n", 
                    //         __LINE__, queryLengthInLastTile0, queryLengthInLastTile1, outputThreadRank0, outputThreadRank1);
                    // }
                    // state.printState();

                    //output
                    const ScoreType groupmaxLastRow = MathOps::reduce_max(group, make_vec2<ScoreType>(maximumInLastRowTracker0.maximum.x, maximumInLastRowTracker1.maximum.y));
                    // printf("line %d, groupmaxLastRow (%d %d)\n", __LINE__, int(groupmaxLastRow.x), int(groupmaxLastRow.y));
                    if(group.thread_rank() == (maxTiles == numTiles0 ? outputThreadRank0 : outputThreadRank1)){
                        ScoreType temp[numItems];
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            temp[i] = maximumInLastColumnTracker.maxima[i];
                            // printf("(%d %d) ", int(temp[i].x), int(temp[i].y));
                        }
                        // printf("\n");

                        if(maxTiles == numTiles0){
                            const int outputRegIndex = (queryLength0-1) % numItems;
                            // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].x;
                            const ScalarScoreType maxLastColumn = temp[outputRegIndex].x;
                            scoreOutput[alignmentId0] = MathOpsScalar::max(groupmaxLastRow.x, maxLastColumn);
                        }else{
                            if(alignmentId1 < inputData.getNumAlignments()){
                                const int outputRegIndex = (queryLength1-1) % numItems;
                                // const ScalarScoreType maxLastColumn = maximumInLastColumnTracker.maxima[outputRegIndex].y;
                                const ScalarScoreType maxLastColumn = temp[outputRegIndex].y;
                                scoreOutput[alignmentId1] = MathOpsScalar::max(groupmaxLastRow.y, maxLastColumn);
                            }
                        }
                    }
                }
            }

        }
    }







} //namespace semiglobalalignment



#endif