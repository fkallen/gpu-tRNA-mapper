#ifndef SEMIGLOBAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH
#define SEMIGLOBAL_ALIGNMENT_KERNELS_SCORE_ONLY_CUH

#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include "tile_processing.cuh"
#include "state_linear.cuh"
#include "state_affine.cuh"
#include "../substitution_score_provider.cuh"
#include "../util.cuh"
#include "../grid_constant_helper.hpp"

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
        GRID_CONSTANT_SPECIFIER int* const scoreOutput,
        GRID_CONSTANT_SPECIFIER const InputData inputData,
        GRID_CONSTANT_SPECIFIER const SUBMAT* const substmatPtr,
        GRID_CONSTANT_SPECIFIER const ScoringKernelParam<ScoreType> scoring
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
        class InputData,
        class SUBMAT
    >
    __global__
    __launch_bounds__(blocksize,1)
    void alphabet_substitutionmatrix_floatOrInt_multipass_kernel(
        GRID_CONSTANT_SPECIFIER int* const scoreOutput,
        GRID_CONSTANT_SPECIFIER const InputData inputData,
        GRID_CONSTANT_SPECIFIER const SUBMAT* const substmatPtr,
        GRID_CONSTANT_SPECIFIER const ScoringKernelParam<ScoreType> scoring,
        GRID_CONSTANT_SPECIFIER char* const tempStorage,
        GRID_CONSTANT_SPECIFIER const size_t tempBytesPerGroup
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






} //namespace semiglobalalignment



#endif