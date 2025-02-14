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


} //namespace localalignment







#endif