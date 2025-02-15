#ifndef SEMIGLOBAL_ALIGNMENT_KERNELS_STARTENDPOS_CUH
#define SEMIGLOBAL_ALIGNMENT_KERNELS_STARTENDPOS_CUH

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
void alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel(
    GRID_CONSTANT_SPECIFIER int* __restrict__ const scoreOutput,
    GRID_CONSTANT_SPECIFIER int* __restrict__ const queryEndPositions_inclusive,
    GRID_CONSTANT_SPECIFIER int* __restrict__ const subjectEndPositions_inclusive,
    GRID_CONSTANT_SPECIFIER const InputData inputData,
    GRID_CONSTANT_SPECIFIER const SUBMAT* __restrict__ const substmatPtr,
    GRID_CONSTANT_SPECIFIER const ScoringKernelParam<ScoreType> scoring,
    GRID_CONSTANT_SPECIFIER char* __restrict__ const tempStorage,
    GRID_CONSTANT_SPECIFIER const size_t tempBytesPerGroup
){
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);
    static_assert(penaltyType == PenaltyType::Affine);

    static_assert(groupsize >= 4);
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

    constexpr int paddingLetter = alphabetSize;
    // constexpr int relaxChunkSize = 4;
    constexpr float oobscore = OOBScore<float>::get();

    const int gapopenscore = scoring.gapopenscore;
    const int gapextendscore = scoring.gapextendscore;

    extern __shared__ float4 externalSharedMem[];
    SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

    for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
        const int row = i / SUBMAT::numColumns;
        const int col = i % SUBMAT::numColumns;
        shared_substmat.data[row][col] = substmatPtr->data[row][col];
    }
    __syncthreads();

    using MathOps = MathOps<ScoreType>;
    using UpdateMaxInLastColumnOp = UpdateMaxInLastColumn<ScoreType, numItems>;
    using UpdateMaxInLastRowOp = UpdateMaxInLastRow<ScoreType>;
    using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
    using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
    // using State = typename std::conditional<
    //     penaltyType == PenaltyType::Linear,
    //     SemiglobalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>,
    //     SemiglobalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxInLastColumnOp, relaxChunkSize>
    // >::type;

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
        // if(group.thread_rank() < group.size() - 1){
        //     groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
        // }
    };

    for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){
        const auto* query = inputData.getQuery(alignmentId);
        const int queryLength = inputData.getQueryLength(alignmentId);
        const int numTiles = SDIV(queryLength, groupsize * numItems);

        ScoreType scoresF[numItems]{};
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        ScoreType E;

        ScoreType maxObserved_lastcol = oobscore;
        ScoreType maxObserved_lastrow = oobscore;
        int positionOfMaxObserved_lastcol_y = 0;
        int positionOfMaxObserved_lastrow_tileNr = 0;
        int positionOfMaxObserved_lastrow_itemIndex = 0;

        TempStorageDataType tileLastColumnM_E;
        TempStorageDataType leftBorderM_E;

        int subjectLength = 0;
        const std::int8_t* subjectData = nullptr;
        int loadOffsetLimit = 0;
        int subjectLoadOffset = 0;
        int currentLetter;
        char4 current4Letters;
        int tempLoadOffset = 0;
        int tempWriteOffset = 0;

        // #define PRINT_WRITE
        // #define PRINT_LOAD

        auto loadQueryLetters = [&](int tileNr, int (&queryLetters)[numItems]){
            #pragma unroll
            for (int i=0; i < numItems; i++) {
                const int index = tileNr * groupsize * numItems + numItems * group.thread_rank()+i;
                if (index >= queryLength) queryLetters[i] = paddingLetter;
                else queryLetters[i] = query[index]; 
            }
        };

        auto printState = [&](int row){
            #if 0
            if(alignmentId == 0){
                if(group.thread_rank() == 0){
                    printf("row %d, currentLetter %d\n", row, currentLetter);
                }
                group.sync();
                if(group.thread_rank() == 0){
                    printf("M:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(scoresM[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(scoresM[(i) % numItems].x), float(scoresM[(firstItemIndex + i) % numItems].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                if(group.thread_rank() == 0){
                    printf("F:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(scoresF[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(scoresF[(i) % numItems].x), float(scoresF[(firstItemIndex + i) % numItems].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                #ifdef USE_E_PRINTARRAY
                if(group.thread_rank() == 0){
                    printf("E:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(Eprintarray[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(Eprintarray[i].x), float(Eprintarray[i].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                #endif
            }
            #endif
        };

        auto loadNext4Letters = [&](){
            current4Letters = make_char4(paddingLetter, paddingLetter, paddingLetter, paddingLetter);
            if(subjectLoadOffset < loadOffsetLimit){
                current4Letters.x = subjectData[subjectLoadOffset];
            }
            if(subjectLoadOffset+1 < loadOffsetLimit){
                current4Letters.y = subjectData[subjectLoadOffset+1];
            }
            if(subjectLoadOffset+2 < loadOffsetLimit){
                current4Letters.z = subjectData[subjectLoadOffset+2];
            }
            if(subjectLoadOffset+3 < loadOffsetLimit){
                current4Letters.w = subjectData[subjectLoadOffset+3];
            }
            subjectLoadOffset += 4*group.size();
        };

        auto shuffleCurrentLetter = [&](){
            currentLetter = group.shfl_up(currentLetter, 1);
        };

        auto shuffle4Letters = [&](){
            static_assert(sizeof(char4) == sizeof(int));
            int temp;
            memcpy(&temp, &current4Letters, sizeof(char4));
            temp = group.shfl_down(temp, 1);
            memcpy(&current4Letters, &temp, sizeof(int));
        };

        auto setTileLastColumn = [&](){
            if(group.thread_rank() == group.size() - 1){
                tileLastColumnM_E.x = scoresM[numItems-1];
                tileLastColumnM_E.y = E;
            }
        };

        auto shuffleTileLastColumn = [&](){
            tileLastColumnM_E = group.shfl_down(tileLastColumnM_E, 1);
        };
        auto shuffleLeftBorder = [&](){
            leftBorderM_E = group.shfl_down(leftBorderM_E, 1);
        };

        auto relaxFirstDiagonal = [&](int row, int tileNr, const auto& substitutionProvider, auto& updateMaxInLastCol){
            static_assert(numItems % 4 == 0);

            ScoreType fooArray[4];
            substitutionProvider.loadFour(group, fooArray, currentLetter, 0);

            //in the first tile E is always computed. In succeeding tiles, E is already computed for the first thread (loaded from temp storage)
            if(tileNr == 0){
                E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
            }else{
                if(group.thread_rank() > 0){
                    E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                }
            }

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMaxInLastCol(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < 4; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMaxInLastCol(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/4; i++){
                substitutionProvider.loadFour(group, fooArray, currentLetter, i);

                #pragma unroll
                for(int k = 0; k < 4; k++){
                    const int index = 4*i + k;
                    E = MathOps::add_max(scoresM[index-1], gapopenscore, MathOps::add(E, gapextendscore));
                    scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore));
                    ScoreType upTempScore = scoresM[index];
                    scoresM[index] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[index]));
                    updateMaxInLastCol(scoresM[index], tileNr, row, index);
                    scoreDiag = upTempScore;
                }
            }


            //initialization of 0-th row in dp matrix for thread rank > 0
            if(row - group.thread_rank() == 0){
                #pragma unroll
                for(int i = 0; i < numItems; i++){
                    scoresM[i] = ScoreType{};
                    scoresF[i] = OOBScore<ScoreType>::get();
                }
            }

            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }

            printState(row);
        };

        auto relax = [&](int row, int tileNr, const auto& substitutionProvider, auto& updateMaxInLastCol){
            static_assert(numItems % 4 == 0);

            ScoreType fooArray[4];
            substitutionProvider.loadFour(group, fooArray, currentLetter, 0);

            // E of current column and scoresF of current row are already computed
            // if(alignmentId == 0){
            //     printf("row %d, tid %d, fooArray[0] %3.f, scoreDiag %3.f, E %3.f, scoresF[0] %3.f, scoresF[1] %3.f, scoresF[2] %3.f, scoresF[3] %3.f\n", 
            //     row, threadIdx.x, fooArray[0], scoreDiag, E, scoresF[0], scoresF[1], scoresF[2], scoresF[3]);
            // }
            // if(alignmentId == 0){
            //     printf("row %d, tid %d,  scoreDiag %3.f, E %3.f, fooArray[0] %3.f, fooArray[1] %3.f, fooArray[2] %3.f, fooArray[3] %3.f\n", 
            //     row, threadIdx.x,  scoreDiag, E, fooArray[0], fooArray[1], fooArray[2], fooArray[3]);
            // }
            ScoreType tempM = scoresM[0];
            scoresM[0] = MathOps::add_max(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMaxInLastCol(scoresM[0], tileNr, row, 0);

            E = MathOps::add_max(scoresM[0],gapopenscore, MathOps::add(E, gapextendscore));
            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore)); //this computes F of the next row !
            scoreDiag = tempM;

            #pragma unroll
            for(int i = 1; i < 4; i++){
                tempM = scoresM[i];
                scoresM[i] = MathOps::add_max(scoreDiag, fooArray[i], MathOps::max(E, scoresF[i]));
                updateMaxInLastCol(scoresM[i], tileNr, row, i);
                E = MathOps::add_max(scoresM[i], gapopenscore, MathOps::add(E, gapextendscore));
                scoresF[i] = MathOps::add_max(scoresM[i], gapopenscore, MathOps::add(scoresF[i], gapextendscore)); //this computes F of the next row !
                scoreDiag = tempM;
            }

            #pragma unroll
            for(int k = 1; k < numItems/4; k++){
                substitutionProvider.loadFour(group, fooArray, currentLetter, k);

                #pragma unroll
                for(int i = 0; i < 4; i++){
                    const int index = k*4+i;
                    tempM = scoresM[index];
                    scoresM[index] = MathOps::add_max(scoreDiag, fooArray[i], MathOps::max(E, scoresF[index]));
                    updateMaxInLastCol(scoresM[index], tileNr, row, index);
                    E = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(E, gapextendscore));
                    scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore)); //this computes F of the next row !
                    scoreDiag = tempM;
                }

            }

            if(row - group.thread_rank() == 0){

                #pragma unroll
                for(int k = 0; k < numItems; k++){
                    scoresM[k] = ScoreType{};
                    #if 0
                    scoresF[k] = OOBScore<ScoreType>::get();
                    //advance F by 1 row to allow for optimized computations of remaining diagonals. (E not important, will get valid E from left neighbor, right neighbor is still OOB)
                    scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                    #else
                    //advance F by 1 row to allow for optimized computations of remaining diagonals. (E not important, will get valid E from left neighbor, right neighbor is still OOB)
                    //since we are in the initialization row, we immediately know the values of F in the next row
                    scoresF[k] = gapopenscore;

                    #endif
                }
            }


            printState(row);
        };

        auto initScoresFirstTile = [&](){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = 0;
                    scoresF[i] = oobscore;
                }
                scoreDiag = 0;
                scoreLeft = 0;
                E = oobscore;
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? 0 : oobscore;
                E = oobscore;
            }
        };

        auto shuffleScoresFirstTile = [&](){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                //scoreLeft is only modified in this function and is initialized with 0 for thread 0
                // assert(scoreLeft == 0);
                //scoreLeft = 0;

                // E = oobscore;
                E = gapopenscore; // After first diagonal was processed, thread 0 needs E of matrix column 1, not -infty
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        };

        auto initScoresNotFirstTile = [&](int tileNr){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = 0;
                    scoresF[i] = oobscore;
                }
                scoreDiag = 0;
                scoreLeft = leftBorderM_E.x;
                E = leftBorderM_E.y;
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? 0 : oobscore;
                E = oobscore;
            }
        };

        auto shuffleScoresNotFirstTile = [&](){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM_E.x;
                E = leftBorderM_E.y;
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        };

        if(numTiles == 1){
            constexpr int tileNr = 0;

            int queryLetters[numItems];
            loadQueryLetters(tileNr, queryLetters);
            SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

            subjectLength = inputData.getSubjectLength(alignmentId);
            subjectData = inputData.getSubject(alignmentId);
            clearOutOfTileTempStorage(subjectLength);

            // if(group.thread_rank() == 0){
            //     printf("subjectLength %d, queryLength %d, globalIndex %d, offset %lu\n", subjectLength, queryLength, globalIndex, charOffset);
            // }
            // group.sync();
            
            loadOffsetLimit = subjectLength;
            subjectLoadOffset = 4*group.thread_rank();
            loadNext4Letters();
            currentLetter = paddingLetter;

            tempWriteOffset = group.thread_rank();

            initScoresFirstTile();

            auto trackLastRow = [&](int row){
                if(row - group.thread_rank() == subjectLength){
                    const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                    #pragma unroll
                    for(int i = 0; i < numItems; i++){
                        const int position = positionOffset + i;
                        if(position < queryLength){
                            if(maxObserved_lastrow < scoresM[i]){
                                maxObserved_lastrow = scoresM[i];
                                positionOfMaxObserved_lastrow_itemIndex = i;
                                positionOfMaxObserved_lastrow_tileNr = tileNr;
                            }
                        }
                    }            
                }
            };

            auto trackLastColumn = [&](ScoreType toCompare, int /*tileNr*/, int row, int itemIndex){
                const int columnOffsetBase = group.thread_rank() * numItems;
                const int columnOffset = columnOffsetBase + itemIndex;
                if(columnOffset == queryLength - 1){
                    if(maxObserved_lastcol < toCompare){
                        maxObserved_lastcol = toCompare;
                        positionOfMaxObserved_lastcol_y = row;
                    }
                }
            };

            const int outputThreadRank = (queryLength-1) / numItems;
            const int numRows = subjectLength + outputThreadRank + 1;
            int r = 1;

            //process first groupsize - 1 diagonals which contain out-of-bound threads
            {
                if(r < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relaxFirstDiagonal(r, tileNr, substitutionProvider, trackLastColumn); //x
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    r++;
                }
                
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    r++;
                }
                
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    r++;
                }
                
                for(; r < min(group.size(), numRows);){
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }

                    if(r < numRows){
                        shuffleCurrentLetter();
                        shuffle4Letters();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //x
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }

                    if(r < numRows){
                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }

                    if(r < numRows){
                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                }
            }

            // process rows which do not cover the last valid row. no lastRowCallback required
            for(; r < numRows - int(group.size()) - 3; r += 4){

                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                shuffleScoresFirstTile();
                //trackLastRow(r);
                
                shuffleCurrentLetter();
                if((r) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    shuffle4Letters();
                }
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                shuffleScoresFirstTile();
                //trackLastRow(r+1);

                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                shuffleScoresFirstTile();
                //trackLastRow(r+2);

                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                shuffleScoresFirstTile();
                //trackLastRow(r+3);
            }

            //process remaining wavefronts which cover the last row
            for(; r < numRows - 3; r += 4){
                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                shuffleScoresFirstTile();
                trackLastRow(r);
                
                shuffleCurrentLetter();
                if((r) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    shuffle4Letters();
                }
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                shuffleScoresFirstTile();
                trackLastRow(r+1);

                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                shuffleScoresFirstTile();
                trackLastRow(r+2);

                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                shuffleScoresFirstTile();
                trackLastRow(r+3);
            }

            //can have at most 3 remaining rows
            if(r < numRows){
                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                shuffleScoresFirstTile();
                trackLastRow(r);
                r++;
            }
            if(r < numRows){
                shuffleCurrentLetter();
                if((r-1) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    shuffle4Letters();
                }
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r, tileNr, substitutionProvider, trackLastColumn); //x                 
                shuffleScoresFirstTile();
                trackLastRow(r);
                r++;
            }
            if(r < numRows){
                shuffleCurrentLetter();
                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                trackLastRow(r);
                r++;
            }

            __builtin_assume(numTiles == 1);
            __builtin_assume(positionOfMaxObserved_lastrow_tileNr == 0);
            const int queryLengthInLastTile = queryLength - (numTiles-1) * (groupsize * numItems);
            const int threadIdOfLastColumn = (queryLengthInLastTile-1) / numItems;

            // printf("thread %d before reduce, maxObserved_lastcol %d, positionOfMaxObserved_lastcol_y %d"
            //     ", maxObserved_lastrow %d, positionOfMaxObserved_lastrow_tileNr %d, positionOfMaxObserved_lastrow_itemIndex %d"
            //     "\n",
            //     threadIdx.x, int(maxObserved_lastcol), positionOfMaxObserved_lastcol_y,
            //     int(maxObserved_lastrow), positionOfMaxObserved_lastrow_tileNr, positionOfMaxObserved_lastrow_itemIndex
            // );

            maxObserved_lastcol = group.shfl(maxObserved_lastcol, threadIdOfLastColumn);
            positionOfMaxObserved_lastcol_y = group.shfl(positionOfMaxObserved_lastcol_y, threadIdOfLastColumn);
            int positionOfMaxObserved_lastrow = positionOfMaxObserved_lastrow_tileNr * groupsize * numItems + group.thread_rank() * numItems + positionOfMaxObserved_lastrow_itemIndex;
            const int2 packed = make_int2(maxObserved_lastrow, positionOfMaxObserved_lastrow);
            const int2 maxPacked = cooperative_groups::reduce(group, packed, [](int2 l, int2 r){
                //score
                if(l.x > r.x) return l;
                if(l.x < r.x) return r;
                //prefer smaller queryEnd
                if(l.y < r.y){
                    return l;
                }else{
                    return r;
                }
            });

            maxObserved_lastrow = maxPacked.x;
            positionOfMaxObserved_lastrow = maxPacked.y;

            // printf("thread %d after reduce, maxObserved_lastcol %d, positionOfMaxObserved_lastcol_y %d"
            //     ", maxObserved_lastrow %d, positionOfMaxObserved_lastrow_tileNr %d, positionOfMaxObserved_lastrow_itemIndex %d, positionOfMaxObserved_lastrow %d"
            //     "\n",
            //     threadIdx.x, int(maxObserved_lastcol), positionOfMaxObserved_lastcol_y,
            //     int(maxObserved_lastrow), positionOfMaxObserved_lastrow_tileNr, positionOfMaxObserved_lastrow_itemIndex, positionOfMaxObserved_lastrow
            // );

            int subjectEndIncl = subjectLength-1;
            int queryEndIncl = queryLength-1;
            int maxObserved = 0;
            if(maxObserved_lastrow > maxObserved_lastcol){
                queryEndIncl = positionOfMaxObserved_lastrow;
                maxObserved = maxObserved_lastrow;
            }else{
                subjectEndIncl = positionOfMaxObserved_lastcol_y - threadIdOfLastColumn - 1;
                maxObserved = maxObserved_lastcol;
            }

            // printf("thread %d before output, maxObserved %d, queryEndIncl %d, subjectEndIncl %d"
            //     "\n",
            //     threadIdx.x, maxObserved, queryEndIncl, subjectEndIncl
            // );

            if(group.thread_rank() == 0){
                scoreOutput[alignmentId] = maxObserved;
                queryEndPositions_inclusive[alignmentId] = queryEndIncl;
                subjectEndPositions_inclusive[alignmentId] = subjectEndIncl;
            }

        }else{

            //first tile
            {
                /* 
                    -----------------------
                    Process tile 0
                    ----------------------- 
                */
                constexpr int tileNr = 0;

                int queryLetters[numItems];
                loadQueryLetters(tileNr, queryLetters);
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

                subjectLength = inputData.getSubjectLength(alignmentId);
                subjectData = inputData.getSubject(alignmentId);
                clearOutOfTileTempStorage(subjectLength);

                // if(group.thread_rank() == 0){
                //     printf("subjectLength %d, queryLength %d, globalIndex %d, offset %lu\n", subjectLength, queryLength, globalIndex, charOffset);
                // }
                // group.sync();
                
                loadOffsetLimit = subjectLength;
                subjectLoadOffset = 4*group.thread_rank();
                loadNext4Letters();
                currentLetter = paddingLetter;

                tempWriteOffset = group.thread_rank();

                initScoresFirstTile();

                auto trackLastRow = [&](int row){
                    if(row - group.thread_rank() == subjectLength){
                        const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            const int position = positionOffset + i;
                            if(position < queryLength){
                                if(maxObserved_lastrow < scoresM[i]){
                                    maxObserved_lastrow = scoresM[i];
                                    positionOfMaxObserved_lastrow_itemIndex = i;
                                    positionOfMaxObserved_lastrow_tileNr = tileNr;
                                }
                            }
                        }            
                    }
                };

                auto trackLastColumn = [&](ScoreType toCompare, int tileNr, int row, int itemIndex){
                    if(tileNr == numTiles-1){
                        const int columnOffsetBase = tileNr * groupsize * numItems + group.thread_rank() * numItems;
                        const int columnOffset = columnOffsetBase + itemIndex;
                        if(columnOffset == queryLength - 1){
                            if(maxObserved_lastcol < toCompare){
                                maxObserved_lastcol = toCompare;
                                positionOfMaxObserved_lastcol_y = row;
                            }
                        }
                    }
                };

                const int numRows = (subjectLength + 1) + (groupsize-1);
                int r = 1;

                //process first groupsize - 1 diagonals which contain out-of-bound threads
                {
                    if(r < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relaxFirstDiagonal(r, tileNr, substitutionProvider, trackLastColumn); //x
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                        shuffleScoresFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    for(; r < min(group.size(), numRows);){
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                            shuffleScoresFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter();
                            shuffle4Letters();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //x
                            shuffleScoresFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                            shuffleScoresFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                            shuffleScoresFirstTile();
                            trackLastRow(r);
                            r++;
                        }
                    }
                }

                // process rows which do not cover the last valid row. no lastRowCallback required
                for(; r < numRows - int(group.size()) - 3; r += 4){

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    //trackLastRow(r);
                    
                    shuffleCurrentLetter();
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    //trackLastRow(r+1);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    //trackLastRow(r+2);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    //trackLastRow(r+3);

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    } 
                }

                //process remaining wavefronts which cover the last row
                for(; r < numRows - 3; r += 4){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    
                    shuffleCurrentLetter();
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r+1);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r+2);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r+3);

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    } 
                }

                //can have at most 3 remaining rows
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //x                 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    trackLastRow(r);
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    trackLastRow(r);
                    r++;
                }

                const int totalChunksOfFour = subjectLength / 4;
                const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + subjectLength % 4;
                if(numThreadsWithValidTileLastColumn > 0){
                    const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                    if(group.thread_rank() >= firstValidThread){
                        #ifdef PRINT_WRITE
                        printf("last write. tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset - firstValidThread);
                        #endif
                        groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                    }
                }

            }



            for(int tileNr = 1; tileNr < numTiles; tileNr++){
                /* 
                    -----------------------
                    Process tile tileNr
                    ----------------------- 
                */

                int queryLetters[numItems];
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                loadQueryLetters(tileNr, queryLetters);

                subjectLoadOffset = 4*group.thread_rank();
                loadNext4Letters();
                currentLetter = paddingLetter;

                tempWriteOffset = group.thread_rank();

                #ifdef PRINT_LOAD
                printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[group.thread_rank()].x, groupTempStorage[group.thread_rank()].y, group.thread_rank());
                #endif
                leftBorderM_E = groupTempStorage[group.thread_rank()];
                tempLoadOffset = group.size() + group.thread_rank();


                initScoresNotFirstTile(tileNr);

                auto trackLastRow = [&](int row){
                    if(row - group.thread_rank() == subjectLength){
                        const int positionOffset = tileNr * group.size() * numItems + group.thread_rank() * numItems;
                        #pragma unroll
                        for(int i = 0; i < numItems; i++){
                            const int position = positionOffset + i;
                            if(position < queryLength){
                                if(maxObserved_lastrow < scoresM[i]){
                                    maxObserved_lastrow = scoresM[i];
                                    positionOfMaxObserved_lastrow_itemIndex = i;
                                    positionOfMaxObserved_lastrow_tileNr = tileNr;
                                }
                            }
                        }            
                    }
                };

                auto trackLastColumn = [&](ScoreType toCompare, int tileNr, int row, int itemIndex){
                    if(tileNr == numTiles-1){
                        const int columnOffsetBase = tileNr * groupsize * numItems + group.thread_rank() * numItems;
                        const int columnOffset = columnOffsetBase + itemIndex;
                        if(columnOffset == queryLength - 1){
                            if(maxObserved_lastcol < toCompare){
                                maxObserved_lastcol = toCompare;
                                positionOfMaxObserved_lastcol_y = row;
                            }
                        }
                    }
                };

                const int numRows = [&](){
                    if(tileNr < numTiles - 1){
                        return (subjectLength + 1) + (groupsize-1);
                    }else{
                        const int queryLengthInLastTile = queryLength - (numTiles-1) * (groupsize * numItems);
                        const int outputThreadRank = (queryLengthInLastTile-1) / numItems;
                        return subjectLength + outputThreadRank + 1;
                    }
                }();
                int r = 1;

                //process first groupsize - 1 diagonals which contain out-of-bound threads
                {
                    if(r < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relaxFirstDiagonal(r, tileNr, substitutionProvider, trackLastColumn); //x
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        trackLastRow(r);
                        r++;
                    }
                    
                    for(; r < min(group.size(), numRows);){
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter();
                            shuffle4Letters();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //x
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile();
                            trackLastRow(r);
                            r++;
                        }
    
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                            relax(r, tileNr, substitutionProvider, trackLastColumn); //z
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile();
                            trackLastRow(r);
                            r++;
                        }
                    }
                }

                // process rows which do not cover the last valid row. no lastRowCallback required
                for(; r < numRows - int(group.size()) - 3; r += 4){

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    if(r % group.size() == 0 && r < subjectLength){
                        #ifdef PRINT_LOAD
                        printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                        #endif
                        leftBorderM_E = groupTempStorage[tempLoadOffset];
                        tempLoadOffset += group.size();
                    }else{
                        shuffleLeftBorder(); //must be called before shuffleScores
                    }
                    shuffleScoresNotFirstTile();
                    //trackLastRow(r);
                    
                    shuffleCurrentLetter();
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    //trackLastRow(r+1);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    //trackLastRow(r+2);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    //trackLastRow(r+3);

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    } 
                }

                //process remaining wavefronts which cover the last row
                for(; r < numRows - 3; r += 4){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    if(r % group.size() == 0 && r < subjectLength){
                        #ifdef PRINT_LOAD
                        printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                        #endif
                        leftBorderM_E = groupTempStorage[tempLoadOffset];
                        tempLoadOffset += group.size();
                    }else{
                        shuffleLeftBorder(); //must be called before shuffleScores
                    }
                    shuffleScoresNotFirstTile();
                    trackLastRow(r);
                    
                    shuffleCurrentLetter();
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider, trackLastColumn); //x
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    trackLastRow(r+1);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    trackLastRow(r+2);

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider, trackLastColumn); //z 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    trackLastRow(r+3);

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    } 
                }

                //can have at most 3 remaining rows
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //w
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    if(r % group.size() == 0 && r < subjectLength){
                        #ifdef PRINT_LOAD
                        printf("last load. tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                        #endif
                        leftBorderM_E = groupTempStorage[tempLoadOffset];
                        tempLoadOffset += group.size();
                    }else{
                        shuffleLeftBorder(); //must be called before shuffleScores
                    }
                    shuffleScoresNotFirstTile();
                    trackLastRow(r);
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //x                 
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    trackLastRow(r);
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r, tileNr, substitutionProvider, trackLastColumn); //y
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    trackLastRow(r);
                    r++;
                }

                const int totalChunksOfFour = subjectLength / 4;
                const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + subjectLength % 4;
                if(numThreadsWithValidTileLastColumn > 0){
                    const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                    if(group.thread_rank() >= firstValidThread){
                        #ifdef PRINT_WRITE
                        printf("last write. tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset - firstValidThread);
                        #endif
                        groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                    }
                }
                
            }
            //printState(r+3);


            const int queryLengthInLastTile = queryLength - (numTiles-1) * (groupsize * numItems);
            const int threadIdOfLastColumn = (queryLengthInLastTile-1) / numItems;

            maxObserved_lastcol = group.shfl(maxObserved_lastcol, threadIdOfLastColumn);
            positionOfMaxObserved_lastcol_y = group.shfl(positionOfMaxObserved_lastcol_y, threadIdOfLastColumn);
            int positionOfMaxObserved_lastrow = positionOfMaxObserved_lastrow_tileNr * groupsize * numItems + group.thread_rank() * numItems + positionOfMaxObserved_lastrow_itemIndex;
            const int2 packed = make_int2(maxObserved_lastrow, positionOfMaxObserved_lastrow);
            const int2 maxPacked = cooperative_groups::reduce(group, packed, [](int2 l, int2 r){
                //score
                if(l.x > r.x) return l;
                if(l.x < r.x) return r;
                //prefer smaller queryEnd
                if(l.y < r.y){
                    return l;
                }else{
                    return r;
                }
            });

            maxObserved_lastrow = maxPacked.x;
            positionOfMaxObserved_lastrow = maxPacked.y;

            int subjectEndIncl = subjectLength-1;
            int queryEndIncl = queryLength-1;
            int maxObserved = 0;
            if(maxObserved_lastrow > maxObserved_lastcol){
                queryEndIncl = positionOfMaxObserved_lastrow;
                maxObserved = maxObserved_lastrow;
            }else{
                subjectEndIncl = positionOfMaxObserved_lastcol_y - threadIdOfLastColumn - 1;
                maxObserved = maxObserved_lastcol;
            }

            if(group.thread_rank() == 0){
                scoreOutput[alignmentId] = maxObserved;
                queryEndPositions_inclusive[alignmentId] = queryEndIncl;
                subjectEndPositions_inclusive[alignmentId] = subjectEndIncl;
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
void call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel(
    int* d_scoreOutput,
    int* d_queryEndPositions_inclusive,
    int* d_subjectEndPositions_inclusive,
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
    auto kernel = alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel<
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
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
        inputData,
        d_substmatPtr,
        scoring,
        d_temp,
        tileTempBytesPerGroup
    );
    CUDACHECKASYNC;
}











//#define USE_E_PRINTARRAY

#ifdef USE_E_PRINTARRAY
#define EPRINTARRAY_FUNCTIONARG ,Eprintarray
#else
#define EPRINTARRAY_FUNCTIONARG 
#endif


/*
    Need to have this relax as free function instead of lambda. The compiler (12.6) is unable to inline the lambda leading to bad performance
*/
template<class ScoreType, int tileSize, class Group, class SubstitutionScoreProvider, int numItems>
__device__
void relax_backwards_untilScoreMatch_freeFunction(
    Group group, const SubstitutionScoreProvider& substitutionProvider, int reverseQueryLength, int, 
    ScoreType oobscore, ScoreType gapopenscore, ScoreType gapextendscore,
    ScoreType (&scoresM) [numItems], ScoreType (&scoresF) [numItems], ScoreType& E,
    ScoreType& scoreDiag,
    const ScoreType scoreLeft,
    int currentLetter,
    int row, int tileNr, ScoreType    
    #ifdef USE_E_PRINTARRAY
    , ScoreType (&Eprintarray) [numItems]
    #endif
){
    static_assert(numItems % 4 == 0);
    using MathOps = MathOps<ScoreType>;

    ScoreType fooArray[4];
    substitutionProvider.loadFour(group, fooArray, currentLetter, 0);

    #pragma unroll
    for(int x = 0; x < 4; x++){
        const int tileColumnIndex = group.thread_rank() * numItems + 0 * 4 + x;
        const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;
        if(globalColumnIndex >= reverseQueryLength){
            fooArray[x] = oobscore;
        }
    }

    // E of current column and scoresF of current row are already computed

    // if(ifprint()){
    //     printf("scoreDiag %3.f\n", )
    // }

    ScoreType tempM = scoresM[0];
    scoresM[0] = MathOps::add_max(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
    E = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(E, gapextendscore));
    scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0],gapextendscore)); //this computes F of the next row !
    scoreDiag = tempM;
    #ifdef USE_E_PRINTARRAY
    Eprintarray[0] = E;
    #endif

    #pragma unroll
    for(int k = 1; k < 4; k++){
        tempM = scoresM[k];
        scoresM[k] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
        E = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(E, gapextendscore));
        scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore)); //this computes F of the next row !
        scoreDiag = tempM;
        #ifdef USE_E_PRINTARRAY
        Eprintarray[k] = E;
        #endif
    }

    #pragma unroll
    for(int i = 1; i < numItems/4; i++){
        substitutionProvider.loadFour(group, fooArray, currentLetter, i);

        #pragma unroll
        for(int x = 0; x < 4; x++){
            const int tileColumnIndex = group.thread_rank() * numItems + i * 4 + x;
            const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;
            if(globalColumnIndex >= reverseQueryLength){
                fooArray[x] = oobscore;
            }
        }
        
        #pragma unroll
        for(int k = 0; k < 4; k++){
            const int index = i*4+k;
            tempM = scoresM[index];
            scoresM[index] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[index]));
            E = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(E, gapextendscore));
            scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore)); //this computes F of the next row !
            scoreDiag = tempM;
            #ifdef USE_E_PRINTARRAY
            Eprintarray[index] = E;
            #endif
        }

        //initialization of 0-th row in dp matrix for thread rank > 0
        if(row - group.thread_rank() == 0){
            scoresM[0] = MathOps::add(scoreLeft, gapextendscore);

            #pragma unroll
            for(int k = 1; k < numItems; k++){
                scoresM[k] = MathOps::add(scoresM[k-1], gapextendscore);
            }

            #pragma unroll
            for(int k = 0; k < numItems; k++){
                #if 0
                scoresF[k] = OOBScore<ScoreType>::get();
                //advance F by 1 row to allow for optimized computations of remaining diagonals. (E not important, will get valid E from left neighbor, right neighbor is still OOB)
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                #else
                //advance F by 1 row to allow for optimized computations of remaining diagonals. (E not important, will get valid E from left neighbor, right neighbor is still OOB)
                //since we are in the initialization row, we know scoresF[k] is currently oob
                scoresF[k] = MathOps::add(scoresM[k], gapopenscore);

                #endif
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
__global__
__launch_bounds__(blocksize,1)
void alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_backwardpass_kernel(
    GRID_CONSTANT_SPECIFIER int* __restrict__ const queryStartPositions_inclusive,
    GRID_CONSTANT_SPECIFIER int* __restrict__ const subjectStartPositions_inclusive,
    GRID_CONSTANT_SPECIFIER const int* __restrict__ const scores,
    GRID_CONSTANT_SPECIFIER const int* __restrict__ const queryEndPositions_inclusive,
    GRID_CONSTANT_SPECIFIER const int* __restrict__ const subjectEndPositions_inclusive,
    GRID_CONSTANT_SPECIFIER const InputData inputData,
    GRID_CONSTANT_SPECIFIER const SUBMAT* __restrict__ const substmatPtr,
    GRID_CONSTANT_SPECIFIER const ScoringKernelParam<ScoreType> scoring,
    GRID_CONSTANT_SPECIFIER char* __restrict__ const tempStorage,
    GRID_CONSTANT_SPECIFIER const size_t tempBytesPerGroup
){
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);
    static_assert(penaltyType == PenaltyType::Affine);

    static_assert(groupsize >= 4);
    static_assert(groupsize <= 32);
    static_assert(blocksize % groupsize == 0);

    constexpr int expectedNumColumnsSUBMAT = alphabetSize;
    constexpr int expectedNumRowsSUBMAT = alphabetSize;

    static_assert(expectedNumRowsSUBMAT == SUBMAT::numRows);
    static_assert(expectedNumColumnsSUBMAT == SUBMAT::numColumns);

    constexpr int tileSize = groupsize * numItems;

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
    // constexpr int relaxChunkSize = 4;
    constexpr float oobscore = OOBScore<float>::get();

    const int gapopenscore = scoring.gapopenscore;
    const int gapextendscore = scoring.gapextendscore;

    extern __shared__ float4 externalSharedMem[];
    SUBMAT& shared_substmat = *((SUBMAT*)((char*)&externalSharedMem[0]));

    for(int i = threadIdx.x; i < SUBMAT::numRows * SUBMAT::numColumns; i += blockDim.x){
        const int row = i / SUBMAT::numColumns;
        const int col = i % SUBMAT::numColumns;
        shared_substmat.data[row][col] = substmatPtr->data[row][col];
    }
    __syncthreads();

    #define BACKWARD_EARLY_EXIT_BY_ROW

    using MathOps = MathOps<ScoreType>;
    using UpdateMaxOp = UpdateMax<ScoreType>;
    using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
    using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
    // using State = typename std::conditional<
    //     penaltyType == PenaltyType::Linear,
    //     SemiglobalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
    //     SemiglobalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
    // >::type;

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

    // TempStorageDataType* groupTempStorageEnd = (TempStorageDataType*)(((char*)tempStorage) + tempBytesPerGroup * (groupIdInGrid+1));
    auto clearOutOfTileTempStorage = [&](int subjectLength, int alignmentId){
    };

    for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){
        const auto* query = inputData.getQuery(alignmentId);
        const int queryEndIncl = queryEndPositions_inclusive[alignmentId];
        const int subjectEndIncl = subjectEndPositions_inclusive[alignmentId];
        const int reverseSubjectLength = subjectEndIncl + 1;
        const int reverseQueryLength = queryEndIncl + 1;

        const int numTiles = SDIV(reverseQueryLength, groupsize * numItems);
        // if(group.thread_rank() == 0){
        //     printf("reverseSubjectLength %d, reverseQueryLength %d, numTiles %d, groupsize %d, numItems %d\n", reverseSubjectLength, reverseQueryLength, numTiles, groupsize, numItems);
        // }

        const ScoreType maxObserved = scores[alignmentId];
        ScoreType scoresF[numItems]{};
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        ScoreType E;
        #ifdef USE_E_PRINTARRAY
        ScoreType Eprintarray[numItems];
        #endif

        int positionOfMaxObserved_lastcol_y = -1;
        int positionOfMaxObserved_lastrow_tileNr = 0;
        int positionOfMaxObserved_lastrow_itemIndex = -1;
        bool groupFoundScore = false;

        const std::int8_t* subjectData = nullptr;

        int loadOffsetLimit = 0;
        int subjectLoadOffset = 0;
        char4 current4Letters;
        int currentLetter = paddingLetter;

        TempStorageDataType tileLastColumnM_E;
        TempStorageDataType leftBorderM_E;
        int tempLoadOffset = 0;
        int tempWriteOffset = 0;

        auto ifprint = [&](){
            // return alignmentId == 0;
            return false;
            // return groupIdInGrid == 8832;
        };


        auto ifprocess = [&](){
            //return iteration == 0 || (alignmentId == 92800 && iteration == 1);
            //return iteration == 0 || (blockIdx.x == 69);
            //return iteration == 1;


            //return iteration == 1 && (68 <= blockIdx.x && blockIdx.x <= 69); //not found
            // return iteration == 1 
            //     && ( (blockIdx.x == 68 ) 
            //         || (blockIdx.x == 69) ); //???

            // return (92800 -100 <= alignmentId) && (alignmentId <= 92800 +100);
            return true;
            // return (groupIdInGrid == 8832);
            // return (8832 -1000 <= groupIdInGrid) && (groupIdInGrid <= 8832 +1000);
        };


        // #define PRINT_WRITE
        // #define PRINT_LOAD

        auto printState = [&](int row){
            #if 0
                //if(group.thread_rank() == 0){
                    printf("row %d, currentLetter %d, tid %d\n", row, currentLetter, group.thread_rank());
                //}
                group.sync();
                if(group.thread_rank() == 0){
                    printf("M:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(scoresM[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(scoresM[(i) % numItems].x), float(scoresM[(firstItemIndex + i) % numItems].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                if(group.thread_rank() == 0){
                    printf("F:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(scoresF[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(scoresF[(i) % numItems].x), float(scoresF[(firstItemIndex + i) % numItems].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                #ifdef USE_E_PRINTARRAY
                if(group.thread_rank() == 0){
                    printf("E:\n");
                }
                group.sync();
                for(int t = group.size()-1; t >= 0; t--){
                    if(group.thread_rank() == t){
                        for(int i = 0; i < t*numItems; i++){
                            printf("    ");
                        }
                        for(int i = 0; i < numItems; i++){
                            printf("%3d ", int(Eprintarray[(i) % numItems]));
                            // printf("(%3.0f %3.0f)", float(Eprintarray[i].x), float(Eprintarray[i].y));
                        }
                        printf("\n");
                    }
                    group.sync();
                }
                #endif
            #endif
        };

        auto needToProcessTile = [&](int tileNr){
            //if the group found the score in the previous iteration and exited early, don't process tile (the right column temp storage may be in an invalid state)
            if(groupFoundScore) return false;

            return true;
        };

        auto loadQueryLetters_reversed = [&](int tileNr, int (&queryLetters)[numItems]){
            #pragma unroll
            for (int i=0; i < numItems; i++) {
                const int index = reverseQueryLength - 1 - (tileNr * groupsize * numItems + numItems * group.thread_rank()+i);
                if (index < 0) queryLetters[i] = paddingLetter;
                else queryLetters[i] = query[index]; 
            }
        };

        auto loadNext4Letters_reversed = [&](){
            current4Letters = make_char4(paddingLetter, paddingLetter, paddingLetter, paddingLetter);
            if(subjectLoadOffset < loadOffsetLimit){
                current4Letters.x = subjectData[reverseSubjectLength - 1 - (subjectLoadOffset)];
            }
            if(subjectLoadOffset+1 < loadOffsetLimit){
                current4Letters.y = subjectData[reverseSubjectLength - 1 - (subjectLoadOffset+1)];
            }
            if(subjectLoadOffset+2 < loadOffsetLimit){
                current4Letters.z = subjectData[reverseSubjectLength - 1 - (subjectLoadOffset+2)];
            }
            if(subjectLoadOffset+3 < loadOffsetLimit){
                current4Letters.w = subjectData[reverseSubjectLength - 1 - (subjectLoadOffset+3)];
            }
            subjectLoadOffset += 4*group.size();  
        };

        auto shuffleCurrentLetter = [&](){
            currentLetter = group.shfl_up(currentLetter, 1);
        };

        auto shuffle4Letters = [&](){
            static_assert(sizeof(char4) == sizeof(int));
            int temp;
            memcpy(&temp, &current4Letters, sizeof(char4));
            temp = group.shfl_down(temp, 1);
            memcpy(&current4Letters, &temp, sizeof(int));
        };

        auto setTileLastColumn = [&](){
            if(group.thread_rank() == group.size() - 1){
                tileLastColumnM_E.x = scoresM[numItems-1];
                tileLastColumnM_E.y = E;
            }
        };

        auto shuffleTileLastColumn = [&](){
            tileLastColumnM_E = group.shfl_down(tileLastColumnM_E, 1);
        };
        auto shuffleLeftBorder = [&](){
            leftBorderM_E = group.shfl_down(leftBorderM_E, 1);
        };

        //does not keep track of observed global maximum.        
        auto relaxFirstDiagonal_backwards_untilScoreMatch = [&](int row, int tileNr, ScoreType, const auto& substitutionProvider){
            static_assert(numItems % 4 == 0);

            ScoreType fooArray[4];
            substitutionProvider.loadFour(group, fooArray, currentLetter, 0);

            #pragma unroll
            for(int x = 0; x < 4; x++){
                const int tileColumnIndex = group.thread_rank() * numItems + 0 * 4 + x;
                const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;
                if(globalColumnIndex >= reverseQueryLength){
                    fooArray[x] = oobscore;
                }
            }

            //in the first tile E is always computed. In succeeding tiles, E is already computed for the first thread (loaded from temp storage)
            if(tileNr == 0){
                E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
            }else{
                if(group.thread_rank() > 0){
                    E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                }
            }

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            scoreDiag = upTempScore;
            #ifdef USE_E_PRINTARRAY
            Eprintarray[0] = E;
            #endif

            #pragma unroll
            for(int k = 1; k < 4; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                scoreDiag = upTempScore;
                #ifdef USE_E_PRINTARRAY
                Eprintarray[k] = E;
                #endif
            }

            #pragma unroll
            for(int i = 1; i < numItems/4; i++){
                substitutionProvider.loadFour(group, fooArray, currentLetter, i);

                #pragma unroll
                for(int x = 0; x < 4; x++){
                    const int tileColumnIndex = group.thread_rank() * numItems + i * 4 + x;
                    const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;
                    if(globalColumnIndex >= reverseQueryLength){
                        fooArray[x] = oobscore;
                    }
                }

                #pragma unroll
                for(int k = 0; k < 4; k++){
                    const int index = i*4+k;
                    E = MathOps::add_max(scoresM[index-1], gapopenscore, MathOps::add(E, gapextendscore));
                    scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore));
                    ScoreType upTempScore = scoresM[index];
                    scoresM[index] = MathOps::add_max(scoreDiag, fooArray[k], MathOps::max(E, scoresF[index]));
                    scoreDiag = upTempScore;
                    #ifdef USE_E_PRINTARRAY
                    Eprintarray[index] = E;
                    #endif
                }
            }

            //initialization of 0-th row in dp matrix for thread rank > 0
            if(row - group.thread_rank() == 0){
                scoresM[0] = MathOps::add(scoreLeft, gapextendscore);
                scoresF[0] = OOBScore<ScoreType>::get();

                #pragma unroll
                for(int i = 1; i < numItems; i++){
                    scoresM[i] = MathOps::add(scoresM[i-1], gapextendscore);
                    scoresF[i] = OOBScore<ScoreType>::get();
                }
            }

            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }
        };

        auto searchScores_lastRow = [&](int row, int tileNr, ScoreType scoreToFind){

            int indexOfScoreToFind = -1;
            const bool isLastRow = row - group.thread_rank() == reverseSubjectLength;
            if(isLastRow){
                #pragma unroll
                for(int k = 0; k < numItems; k++){
                    const int tileColumnIndex = group.thread_rank() * numItems + k;
                    const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;

                    if(globalColumnIndex < reverseQueryLength){
                        if(scoreToFind == scoresM[k]){
                            indexOfScoreToFind = k;
                            // if(ifprint()){
                            //     printf("found in last row. thread %d, row %d, k %d, scoresM[k] %f\n", 
                            //         threadIdx.x + blockIdx.x * blockDim.x,
                            //         row,
                            //         k,
                            //         scoresM[k]
                            //     );
                            // }
                            break;
                        }
                    }
                }
            }
            if(indexOfScoreToFind != -1){
                positionOfMaxObserved_lastrow_tileNr = tileNr;
                positionOfMaxObserved_lastrow_itemIndex = indexOfScoreToFind;
            }
        };
        auto searchScores_lastCol = [&](int row, int tileNr, ScoreType scoreToFind){
            int indexOfScoreToFind = -1;

            const int columnOffsetBase = tileNr * tileSize + group.thread_rank() * numItems;
            #pragma unroll
            for(int k = 0; k < numItems; k++){
                const int columnOffset = columnOffsetBase + k;

                if(columnOffset == reverseQueryLength - 1){
                    if(scoreToFind == scoresM[k]){
                        indexOfScoreToFind = k;
                        // if(ifprint()){
                        //     printf("found in last col. thread %d, row %d, k %d, scoresM[k] %f\n", 
                        //         threadIdx.x + blockIdx.x * blockDim.x,
                        //         row,
                        //         k,
                        //         scoresM[k]
                        //     );
                        // }
                        break;
                    }
                }
            }
            if(indexOfScoreToFind != -1){
                positionOfMaxObserved_lastcol_y = row - group.thread_rank();
            }
        };

        auto mySearchWasSuccessful = [&](){
            return positionOfMaxObserved_lastcol_y > -1 || positionOfMaxObserved_lastrow_itemIndex > -1;
        };

        //for the backward pass, first row and first col are initialized the same way as a global alignment
        //(reverse alignment should begin at the exact position which we computed in the forward pass, no more free gaps at the begin)

        auto initScoresFirstTile = [&](){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    const int column = 1+(0 * int(group.size()) * numItems) + (numItems * group.thread_rank() + i);
                    scoresM[i] = MathOps::add(gapopenscore, (column-1) * gapextendscore);
                    scoresF[i] = oobscore;
                }
                scoreDiag = ScoreType{};
                scoreLeft = gapopenscore;
                E = oobscore;
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? MathOps::add(gapopenscore, ((0 * int(group.size()) * numItems) + numItems-1) * gapextendscore) : oobscore;
                E = oobscore;
            }
        };

        auto shuffleScoresFirstTile = [&](int row){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                const ScoreType leftBorderM = MathOps::add(gapopenscore, MultiplyWithInt<ScoreType>{}(row, gapextendscore));
                scoreLeft = leftBorderM;
                //subsequent diagonals assume that the first E (E = max(scoreLeft + gapopenscore, E + gapextendscore)) is already computed
                //compute and init the value for thread 0 accordingly. E will be OOB initially so E + gapextendscore does not need to be considered
                E = MathOps::add(leftBorderM, gapopenscore);
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        };

        auto initScoresNotFirstTile = [&](int tileNr){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    const int column = 1+(tileNr * int(group.size()) * numItems) + (numItems * group.thread_rank() + i);
                    scoresM[i] = MathOps::add(gapopenscore, (column-1) * gapextendscore);
                    scoresF[i] = oobscore;
                }
                scoreDiag = MathOps::add(gapopenscore, (tileNr * int(group.size()) * numItems - 1) * gapextendscore);
                scoreLeft = leftBorderM_E.x;
                E = leftBorderM_E.y;
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? MathOps::add(gapopenscore, ((tileNr * int(group.size()) * numItems) + numItems-1) * gapextendscore) : oobscore;
                E = oobscore;
            }
        };

        auto shuffleScoresNotFirstTile = [&](int /*row*/){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM_E.x;
                E = leftBorderM_E.y;
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        };

        

        if(numTiles == 1){
            constexpr int tileNr = 0;
            
            int queryLetters[numItems];
            SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
            subjectData = inputData.getSubject(alignmentId);

            loadOffsetLimit = reverseSubjectLength;
            subjectLoadOffset = 4*group.thread_rank();
            currentLetter = paddingLetter;

            clearOutOfTileTempStorage(reverseSubjectLength, alignmentId);

            loadQueryLetters_reversed(0, queryLetters); //query
            loadNext4Letters_reversed(); //subject
            initScoresFirstTile();
            
            __builtin_assume(numTiles == 1);
            const int queryLengthInLastTile = reverseQueryLength - (numTiles-1) * (groupsize * numItems);
            const int threadIdOfLastColumn = (queryLengthInLastTile-1) / numItems;
            const int numRows = reverseSubjectLength + threadIdOfLastColumn + 1;
            int r = 1;

            if(false && false){
                if(group.thread_rank() == 0){
                    printf("initial state\n");
                }
                printState(0);
            }

            const ScoreType& scoreToFind = maxObserved;

            #ifdef BACKWARD_EARLY_EXIT_BY_ROW
            do{
                //process first groupsize - 1 diagonals which contain out-of-bound threads

                if(r < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }
                    r++;
                }

                for(; r < min(group.size(), numRows);){
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }           
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        shuffle4Letters();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                }
                //break after loop
                if(groupFoundScore) break;

                // process rows which do not cover the last valid row. no lastRowCallback required
                for(; r < numRows - int(group.size()) - 3; r += 4){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }

                    shuffleCurrentLetter(); 
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters_reversed();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                    searchScores_lastCol(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+1);
                    if(false && false){
                        printState(r+1);
                    }

                    shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                    searchScores_lastCol(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+2);
                    if(false && false){
                        printState(r+2);
                    }

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                    searchScores_lastCol(r+3, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+3);
                    if(false && false){
                        printState(r+3);
                    }
                }
                //break after loop
                if(groupFoundScore) break;

                //process remaining wavefronts which cover the last row
                for(; r < numRows - 3; r += 4){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }

                    shuffleCurrentLetter(); 
                    if((r) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters_reversed();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                    searchScores_lastRow(r+1, tileNr, scoreToFind);
                    searchScores_lastCol(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+1);
                    if(false && false){
                        printState(r+1);
                    }

                    shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                    searchScores_lastRow(r+2, tileNr, scoreToFind);
                    searchScores_lastCol(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+2);
                    if(false && false){
                        printState(r+2);
                    }

                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                    searchScores_lastRow(r+3, tileNr, scoreToFind);
                    searchScores_lastCol(r+3, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r+3);         
                    if(false && false){
                        printState(r+3);
                    }
                }
                //break after loop
                if(groupFoundScore) break;



                //can have at most 3 remaining rows
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters_reversed();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;                 
                    shuffleScoresFirstTile(r);
                    if(false && false){
                        printState(r);
                    }
                    r++;
                }
                if(r < numRows){
                    shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                    searchScores_lastRow(r, tileNr, scoreToFind);
                    searchScores_lastCol(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(mySearchWasSuccessful());
                    if(groupFoundScore) break;
                    if(false && false){
                        printState(r);
                    }
                }

            }while(false);
            #else
                if(group.thread_rank() == 0){
                    printf("non-early exit not implemented\n");
                }
            #endif
            if(!groupFoundScore){
                printf("tid %d did not find score. alignmentId %d\n", 
                    threadIdx.x + blockIdx.x * blockDim.x, alignmentId);
            }

            __builtin_assume(positionOfMaxObserved_lastrow_tileNr == 0);
            positionOfMaxObserved_lastcol_y = group.shfl(positionOfMaxObserved_lastcol_y, threadIdOfLastColumn);
            int positionOfMaxObserved_lastrow = positionOfMaxObserved_lastrow_tileNr * groupsize * numItems + group.thread_rank() * numItems + positionOfMaxObserved_lastrow_itemIndex;

            // printf("thread %d before reduce, success %d, positionOfMaxObserved_lastcol_y %d"
            //     ", positionOfMaxObserved_lastrow_tileNr %d, positionOfMaxObserved_lastrow_itemIndex %d, positionOfMaxObserved_lastrow %d"
            //     "\n",
            //     threadIdx.x, mySearchWasSuccessful(), positionOfMaxObserved_lastcol_y,
            //     positionOfMaxObserved_lastrow_tileNr, positionOfMaxObserved_lastrow_itemIndex, positionOfMaxObserved_lastrow
            // );

            const int2 packed = make_int2(positionOfMaxObserved_lastrow_itemIndex > -1, positionOfMaxObserved_lastrow);
            const int2 maxPacked = cooperative_groups::reduce(group, packed, [](int2 l, int2 r){
                //score
                if(l.x > r.x) return l;
                if(l.x < r.x) return r;
                //prefer smaller queryBegin
                if(l.y < r.y){
                    return l;
                }else{
                    return r;
                }
            });
            positionOfMaxObserved_lastrow = maxPacked.y;

            // printf("thread %d after reduce, success %d, positionOfMaxObserved_lastrow %d"
            //     ", reverseQueryLength %d, reverseSubjectLength, %d"    
            //     "\n",
            //     threadIdx.x, maxPacked.x, positionOfMaxObserved_lastrow,
            //     reverseQueryLength, reverseSubjectLength
            // );

            int reverseSubjectEndIncl = reverseSubjectLength-1;
            int reverseQueryEndIncl = reverseQueryLength-1;

            if(positionOfMaxObserved_lastcol_y > -1){
                reverseSubjectEndIncl = positionOfMaxObserved_lastcol_y - 1;
            }else{
                reverseQueryEndIncl = positionOfMaxObserved_lastrow;
            }

            if(group.thread_rank() == 0){
                queryStartPositions_inclusive[alignmentId] = reverseQueryLength - reverseQueryEndIncl - 1;
                subjectStartPositions_inclusive[alignmentId] = reverseSubjectLength - reverseSubjectEndIncl - 1;
            }
        }else{
            const ScoreType& scoreToFind = maxObserved;

            //first tile
            {
                /* 
                    -----------------------
                    Process tile 0
                    ----------------------- 
                */
                constexpr int tileNr = 0;

                int queryLetters[numItems];
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
                subjectData = inputData.getSubject(alignmentId);
    
                loadOffsetLimit = reverseSubjectLength;
                subjectLoadOffset = 4*group.thread_rank();
                currentLetter = paddingLetter;
    
                clearOutOfTileTempStorage(reverseSubjectLength, alignmentId);
    
                loadQueryLetters_reversed(tileNr, queryLetters); //query
                loadNext4Letters_reversed(); //subject
                initScoresFirstTile();

                tempWriteOffset = group.thread_rank();


                const int numRows = (reverseSubjectLength + 1) + (groupsize-1);
                int r = 1;

                #ifdef BACKWARD_EARLY_EXIT_BY_ROW
                do{
                    // NOTE: in a multi-tile setting, the first tile does not cover the last column. 
                    // do not need to attempt checking for last column (searchScores_lastCol)


                    //process first groupsize - 1 diagonals which contain out-of-bound threads

                    if(r < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }

                    for(; r < min(group.size(), numRows);){
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }           
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleScoresFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter();
                            shuffle4Letters();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleScoresFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleScoresFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleScoresFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    // process rows which do not cover the last valid row. no lastRowCallback required
                    for(; r < numRows - int(group.size()) - 3; r += 4){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }

                        shuffleCurrentLetter(); 
                        if((r) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+1);
                        if(false && false){
                            printState(r+1);
                        }

                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+2);
                        if(false && false){
                            printState(r+2);
                        }

                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+3);
                        if(false && false){
                            printState(r+3);
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        } 
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    //process remaining wavefronts which cover the last row
                    for(; r < numRows - 3; r += 4){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }

                        shuffleCurrentLetter(); 
                        if((r) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastRow(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+1);
                        if(false && false){
                            printState(r+1);
                        }

                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+2);
                        if(false && false){
                            printState(r+2);
                        }

                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                        searchScores_lastRow(r+3, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r+3);         
                        if(false && false){
                            printState(r+3);
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        } 
                    }
                    //break after loop
                    if(groupFoundScore) break;



                    //can have at most 3 remaining rows
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if((r-1) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;                 
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleScoresFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(false && false){
                            printState(r);
                        }
                    }

                    const int totalChunksOfFour = reverseSubjectLength / 4;
                    const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                    const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + reverseSubjectLength % 4;
                    if(numThreadsWithValidTileLastColumn > 0){
                        const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                        if(group.thread_rank() >= firstValidThread){
                            #ifdef PRINT_WRITE
                            printf("last write. tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset - firstValidThread);
                            #endif
                            groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                        }
                    }

                }while(false);
                #else
                    if(group.thread_rank() == 0){
                        printf("non-early exit not implemented\n");
                    }
                #endif

            }

            for(int tileNr = 1; tileNr < numTiles; tileNr++){
                /* 
                    -----------------------
                    Process tile tileNr
                    ----------------------- 
                */

                int queryLetters[numItems];
                SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);
    
                subjectLoadOffset = 4*group.thread_rank();
                currentLetter = paddingLetter;
    
                clearOutOfTileTempStorage(reverseSubjectLength, alignmentId);
    
                loadQueryLetters_reversed(tileNr, queryLetters); //query
                loadNext4Letters_reversed(); //subject
                

                tempWriteOffset = group.thread_rank();

                #ifdef PRINT_LOAD
                printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[group.thread_rank()].x, groupTempStorage[group.thread_rank()].y, group.thread_rank());
                #endif
                leftBorderM_E = groupTempStorage[group.thread_rank()];
                tempLoadOffset = group.size() + group.thread_rank();
                initScoresNotFirstTile(tileNr);


                const int numRows = [&](){
                    if(tileNr < numTiles - 1){
                        return (reverseSubjectLength + 1) + (groupsize-1);
                    }else{
                        const int queryLengthInLastTile = reverseQueryLength - (numTiles-1) * (groupsize * numItems);
                        const int outputThreadRank = (queryLengthInLastTile-1) / numItems;
                        return reverseSubjectLength + outputThreadRank + 1;
                    }
                }();
                int r = 1;

                #ifdef BACKWARD_EARLY_EXIT_BY_ROW
                do{
                    //process first groupsize - 1 diagonals which contain out-of-bound threads

                    if(r < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }

                    for(; r < min(group.size(), numRows);){
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }           
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            searchScores_lastCol(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter();
                            shuffle4Letters();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            searchScores_lastCol(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter(); 
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            searchScores_lastCol(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                        if(r < numRows){
                            shuffleCurrentLetter();
                            if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                            relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z
                            searchScores_lastRow(r, tileNr, scoreToFind);
                            searchScores_lastCol(r, tileNr, scoreToFind);
                            groupFoundScore = group.any(mySearchWasSuccessful());
                            if(groupFoundScore) break;
                            shuffleLeftBorder(); //must be called before shuffleScores
                            shuffleScoresNotFirstTile(r);
                            if(false && false){
                                printState(r);
                            }
                            r++;
                        }
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    // process rows which do not cover the last valid row. no lastRowCallback required
                    for(; r < numRows - int(group.size()) - 3; r += 4){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }

                        shuffleCurrentLetter(); 
                        if((r) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastCol(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+1);
                        if(false && false){
                            printState(r+1);
                        }

                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastCol(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+2);
                        if(false && false){
                            printState(r+2);
                        }

                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                        searchScores_lastCol(r+3, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+3);
                        if(false && false){
                            printState(r+3);
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        } 
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    //process remaining wavefronts which cover the last row
                    for(; r < numRows - 3; r += 4){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }

                        shuffleCurrentLetter(); 
                        if((r) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+1, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastRow(r+1, tileNr, scoreToFind);
                        searchScores_lastCol(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+1);
                        if(false && false){
                            printState(r+1);
                        }

                        shuffleCurrentLetter(); 
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+2, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r+2, tileNr, scoreToFind);
                        searchScores_lastCol(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+2);
                        if(false && false){
                            printState(r+2);
                        }

                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r+3, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //z 
                        searchScores_lastRow(r+3, tileNr, scoreToFind);
                        searchScores_lastCol(r+3, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r+3);         
                        if(false && false){
                            printState(r+3);
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        } 
                    }
                    //break after loop
                    if(groupFoundScore) break;



                    //can have at most 3 remaining rows
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //w
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("last load. tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if((r-1) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //x
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;                 
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile(r);
                        if(false && false){
                            printState(r);
                        }
                        r++;
                    }
                    if(r < numRows){
                        shuffleCurrentLetter();
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, scoreLeft, currentLetter, r, tileNr, scoreToFind EPRINTARRAY_FUNCTIONARG); //y
                        searchScores_lastRow(r, tileNr, scoreToFind);
                        searchScores_lastCol(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(mySearchWasSuccessful());
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(false && false){
                            printState(r);
                        }
                    }

                    const int totalChunksOfFour = reverseSubjectLength / 4;
                    const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                    const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + reverseSubjectLength % 4;
                    if(numThreadsWithValidTileLastColumn > 0){
                        const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                        if(group.thread_rank() >= firstValidThread){
                            #ifdef PRINT_WRITE
                            printf("last write. tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset - firstValidThread);
                            #endif
                            groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                        }
                    }

                }while(false);
                #else
                    if(group.thread_rank() == 0){
                        printf("non-early exit not implemented\n");
                    }
                #endif

            }

            if(!groupFoundScore){
                printf("tid %d did not find score. alignmentId %d\n", 
                    threadIdx.x + blockIdx.x * blockDim.x, alignmentId);
            }

            const int queryLengthInLastTile = reverseQueryLength - (numTiles-1) * (groupsize * numItems);
            const int threadIdOfLastColumn = (queryLengthInLastTile-1) / numItems;

            positionOfMaxObserved_lastcol_y = group.shfl(positionOfMaxObserved_lastcol_y, threadIdOfLastColumn);
            int positionOfMaxObserved_lastrow = positionOfMaxObserved_lastrow_tileNr * groupsize * numItems + group.thread_rank() * numItems + positionOfMaxObserved_lastrow_itemIndex;

            // printf("thread %d before reduce, success %d, positionOfMaxObserved_lastcol_y %d"
            //     ", positionOfMaxObserved_lastrow_tileNr %d, positionOfMaxObserved_lastrow_itemIndex %d, positionOfMaxObserved_lastrow %d"
            //     "\n",
            //     threadIdx.x, mySearchWasSuccessful(), positionOfMaxObserved_lastcol_y,
            //     positionOfMaxObserved_lastrow_tileNr, positionOfMaxObserved_lastrow_itemIndex, positionOfMaxObserved_lastrow
            // );

            const int2 packed = make_int2(positionOfMaxObserved_lastrow_itemIndex > -1, positionOfMaxObserved_lastrow);
            const int2 maxPacked = cooperative_groups::reduce(group, packed, [](int2 l, int2 r){
                //score
                if(l.x > r.x) return l;
                if(l.x < r.x) return r;
                //prefer smaller queryBegin
                if(l.y < r.y){
                    return l;
                }else{
                    return r;
                }
            });
            positionOfMaxObserved_lastrow = maxPacked.y;

            // printf("thread %d after reduce, success %d, positionOfMaxObserved_lastrow %d"
            //     "\n",
            //     threadIdx.x, maxPacked.x, positionOfMaxObserved_lastrow
            // );

            int reverseSubjectEndIncl = reverseSubjectLength-1;
            int reverseQueryEndIncl = reverseQueryLength-1;

            if(positionOfMaxObserved_lastcol_y > -1){
                reverseSubjectEndIncl = positionOfMaxObserved_lastcol_y - 1;
            }else{
                reverseQueryEndIncl = positionOfMaxObserved_lastrow;
            }

            if(group.thread_rank() == 0){
                queryStartPositions_inclusive[alignmentId] = reverseQueryLength - reverseQueryEndIncl - 1;
                subjectStartPositions_inclusive[alignmentId] = reverseSubjectLength - reverseSubjectEndIncl - 1;
            }
        }
    }

    #ifdef BACKWARD_EARLY_EXIT_BY_ROW
    #undef BACKWARD_EARLY_EXIT_BY_ROW
    #endif
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
void call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_backwardpass_kernel(
    int* d_queryStartPositions_inclusive,
    int* d_subjectStartPositions_inclusive,
    const int* d_scores,
    const int* d_queryEndPositions_inclusive,
    const int* d_subjectEndPositions_inclusive,
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
    auto kernel = alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_backwardpass_kernel<
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
        std::cout << "Not enough smem available.";
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
        d_queryStartPositions_inclusive,
        d_subjectStartPositions_inclusive,
        d_scores,
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
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