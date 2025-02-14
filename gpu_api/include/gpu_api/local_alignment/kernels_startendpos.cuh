#ifndef LOCAL_ALIGNMENT_KERNELS_STARTENDPOS_CUH
#define LOCAL_ALIGNMENT_KERNELS_STARTENDPOS_CUH

#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include "tile_processing.cuh"
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
void alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel(
    __grid_constant__ int* __restrict__ const scoreOutput,
    __grid_constant__ int* __restrict__ const queryEndPositions_inclusive,
    __grid_constant__ int* __restrict__ const subjectEndPositions_inclusive,
    __grid_constant__ const InputData inputData,
    __grid_constant__ const SUBMAT* __restrict__ const substmatPtr,
    __grid_constant__ const ScoringKernelParam<ScoreType> scoring,
    __grid_constant__ char* __restrict__ const tempStorage,
    __grid_constant__ const size_t tempBytesPerGroup
){
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);
    static_assert(penaltyType == PenaltyType::Affine);

    static_assert(groupsize >= 4);
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
    using UpdateMaxOp = UpdateMax<ScoreType>;
    using SubstitutionScoreProvider = SubstitutionMatrixSubstitutionScoreProvider<SUBMAT, ScoreType, numItems>;
    using SubjectLettersData = SubjectLettersData<decltype(group), paddingLetter>;
    // using State = typename std::conditional<
    //     penaltyType == PenaltyType::Linear,
    //     LocalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
    //     LocalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
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
        if(group.thread_rank() < group.size() - 1){
            groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
        }
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
        ScoreType maxObserved = oobscore;
        int positionOfMaxObserved_y = 0;
        int positionOfMaxObserved_tileNr = 0;
        int positionOfMaxObserved_itemIndex = 0;

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

        auto relaxFirstDiagonal = [&](int row, int tileNr, const auto& substitutionProvider){
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
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < 4; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/4; i++){
                substitutionProvider.loadFour(group, fooArray, currentLetter, i);

                #pragma unroll
                for(int k = 0; k < 4; k++){
                    E = MathOps::add_max(scoresM[4*i + k-1], gapopenscore, MathOps::add(E, gapextendscore));
                    scoresF[4*i + k] = MathOps::add_max(scoresM[4*i + k], gapopenscore, MathOps::add(scoresF[4*i + k], gapextendscore));
                    ScoreType upTempScore = scoresM[4*i + k];
                    scoresM[4*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[4*i + k]));
                    scoreDiag = upTempScore;
                }
            }


            // const int myrow = row - group.thread_rank();
            // if(myrow <= subjectLength){
                #pragma unroll
                for(int k = 0; k < numItems; k++){
                    // const int dpColumnIndex = tileNr * tileSize + group.thread_rank() * numItems+k;
                    // if(dpColumnIndex < queryLength){
                        if(maxObserved < scoresM[k]){
                            maxObserved = scoresM[k];
                            positionOfMaxObserved_tileNr = tileNr;
                            positionOfMaxObserved_itemIndex = k;
                            positionOfMaxObserved_y = row;
                        }
                    // }
                }
            // }


            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }

            printState(row);
        };

        auto relax = [&](int row, int tileNr, const auto& substitutionProvider){
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
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));

            E = MathOps::add_max(scoresM[0],gapopenscore, MathOps::add(E, gapextendscore));
            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore)); //this computes F of the next row !
            scoreDiag = tempM;

            #pragma unroll
            for(int i = 1; i < 4; i++){
                tempM = scoresM[i];
                scoresM[i] = MathOps::add_max_relu(scoreDiag, fooArray[i], MathOps::max(E, scoresF[i]));
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
                    scoresM[index] = MathOps::add_max_relu(scoreDiag, fooArray[i], MathOps::max(E, scoresF[index]));
                    E = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(E, gapextendscore));
                    scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore)); //this computes F of the next row !
                    scoreDiag = tempM;
                }

            }

            // const int myrow = row - group.thread_rank();
            // if(myrow <= subjectLength){
                #pragma unroll
                for(int k = 0; k < numItems; k++){
                    // const int dpColumnIndex = tileNr * tileSize + group.thread_rank() * numItems+k;
                    // if(dpColumnIndex < queryLength){
                        if(maxObserved < scoresM[k]){
                            maxObserved = scoresM[k];
                            positionOfMaxObserved_tileNr = tileNr;
                            positionOfMaxObserved_itemIndex = k;
                            positionOfMaxObserved_y = row;
                        }
                    // }
                }
            // }


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

        //first tile
        {
            /* 
                -----------------------
                Process tile 0
                ----------------------- 
            */
            constexpr int tileNr = 0;

            int queryLetters[numItems];
            loadQueryLetters(0, queryLetters);
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

            const int numRows = (subjectLength + 1) + (groupsize-1);
            int r = 1;

            //process first groupsize - 1 diagonals which contain out-of-bound threads
            {
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relaxFirstDiagonal(r, tileNr, substitutionProvider); //x
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+1, tileNr, substitutionProvider); //y
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+2, tileNr, substitutionProvider); //z
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                shuffle4Letters();

                r = 4;
                for(; r < groupsize - 1; r += 4){                    
                    relax(r, tileNr, substitutionProvider); //w
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider); //x
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider); //y
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider); //z
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();
                }
            }

            //process remaining diagonals. process in chunks of 4 diagonals.
            //for those diagonals we need to store the last column of the tile to temp memory
            //last column is stored in "rightBorder"

            //r starts with r=max(4, groupsize)
            for(; r < numRows - 3; r += 4){

                relax(r, tileNr, substitutionProvider); //w
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleScoresFirstTile();
                shuffleCurrentLetter(); 


                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider); //x
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleScoresFirstTile();
                shuffleCurrentLetter(); 


                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider); //y
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleScoresFirstTile();
                shuffleCurrentLetter();


                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+3, tileNr, substitutionProvider); //z 
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleScoresFirstTile();
                shuffleCurrentLetter(); 

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                if((r + 4) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    shuffle4Letters();
                }

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
                relax(r, tileNr, substitutionProvider); //w
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

            }
            if(r+1 < numRows){
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider); //x
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores                    
                shuffleScoresFirstTile();
                shuffleCurrentLetter();
            }
            if(r+2 < numRows){
                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider); //y
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
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

            const int numRows = (subjectLength + 1) + (groupsize-1);
            int r = 1;

            //process first groupsize - 1 diagonals which contain out-of-bound threads
            {
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relaxFirstDiagonal(r, tileNr, substitutionProvider); //x
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+1, tileNr, substitutionProvider); //y
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+2, tileNr, substitutionProvider); //z
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                shuffle4Letters();

                r = 4;
                for(; r < groupsize - 1; r += 4){                    
                    relax(r, tileNr, substitutionProvider); //w
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax(r+1, tileNr, substitutionProvider); //x
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax(r+2, tileNr, substitutionProvider); //y
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax(r+3, tileNr, substitutionProvider); //z
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();
                }
            }

            //process remaining diagonals. process in chunks of 4 diagonals.
            //for those diagonals we need to store the last column of the tile to temp memory
            //last column is stored in "rightBorder"

            //r starts with r=max(4, groupsize)
            for(; r < numRows - 3; r += 4){

                relax(r, tileNr, substitutionProvider); //w
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
                shuffleCurrentLetter(); 
                



                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider); //x
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter(); 
                

                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider); //y
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter(); 
                

                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax(r+3, tileNr, substitutionProvider); //z
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter(); 

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                if((r + 4) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    shuffle4Letters();
                }

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
                relax(r, tileNr, substitutionProvider); //w
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
                shuffleCurrentLetter();
                

            }
            if(r+1 < numRows){
                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relax(r+1, tileNr, substitutionProvider); //x
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
                shuffleLeftBorder(); //must be called before shuffleScores
                shuffleScoresNotFirstTile();
                shuffleCurrentLetter();
                
            }
            if(r+2 < numRows){
                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax(r+2, tileNr, substitutionProvider); //y
                shuffleTileLastColumn(); //must be called before setTileLastColumn
                setTileLastColumn(); //must be called before shuffleScores
            }

            const int totalChunksOfFour = subjectLength / 4;
            const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
            const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + subjectLength % 4;
            if(numThreadsWithValidTileLastColumn > 0){
                const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                if(group.thread_rank() >= firstValidThread){
                    #ifdef PRINT_WRITE
                    printf("last write. tid %d, write %f %f\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y);
                    #endif
                    groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                }
            }
            
        }
        //printState(r+3);


        // if(positionOfMaxObserved_y - group.thread_rank() - 1 >= subjectLength){
        //     printf("iteration %d, tid %d, block %d, alignmentId %d, subjectLength %d, "
        //     "positionOfMaxObserved_tileNr %d, positionOfMaxObserved_itemIndex %d, positionOfMaxObserved_y %d, maxObserved %d\n", 
        //         iteration, threadIdx.x, blockIdx.x, alignmentId, subjectLength, 
        //         positionOfMaxObserved_tileNr,
        //         positionOfMaxObserved_itemIndex,
        //         positionOfMaxObserved_y, int(maxObserved));
        // }

        //disable any maxima that come from out-of-bounds cells
        //query boundaries
        const int myQueryEndInclusive = positionOfMaxObserved_tileNr * groupsize * numItems + group.thread_rank() * numItems + positionOfMaxObserved_itemIndex;
        const int mySubjectEndInclusive = positionOfMaxObserved_y - group.thread_rank() - 1;
        if(myQueryEndInclusive >= queryLength){
            maxObserved = oobscore;
        }
        if(mySubjectEndInclusive < 0 || mySubjectEndInclusive >= subjectLength){
            maxObserved = oobscore;
        }
        const int3 packed = make_int3(maxObserved, myQueryEndInclusive, mySubjectEndInclusive);
        const int3 maxPacked = cooperative_groups::reduce(group, packed, [](int3 l, int3 r){
            //score
            if(l.x > r.x) return l;
            if(l.x < r.x) return r;
            //prefer smaller queryEnd
            if(l.y < r.y){
                return l;
            }else{
                return r;
            }
            // if(l.y > r.y) return l;
            // if(l.y < r.y) return r;
            // if(l.z > r.z){
            //     return l;
            // }else{
            //     return r;
            // }
        });

        // if(alignmentId == 0){
        //     printf("iteration %d, tid %d, reduction input. x %d, y %d, z %d.\n",
        //         iteration,
        //         threadIdx.x,
        //         int(maxObserved),
        //         myQueryEndInclusive,
        //         mySubjectEndInclusive
        //     );

        //     printf("iteration %d, tid %d, reduction output. x %d, y %d, z %d.\n",
        //         iteration,
        //         threadIdx.x,
        //         maxPacked.x,
        //         maxPacked.y,
        //         maxPacked.z
        //     );
        // }

        const int score = maxPacked.x;
        const int subjectEndIncl = maxPacked.z;
        const int queryEndIncl = maxPacked.y;

        if(subjectEndIncl >= subjectLength || subjectEndIncl < 0){
        // if(alignmentId == 0){
            printf("tid %d, block %d, alignmentId %d, subjectLength %d, "
            "positionOfMaxObserved_tileNr %d, positionOfMaxObserved_itemIndex %d, positionOfMaxObserved_y %d, maxObserved %d, "
            "resultscore %d, subjectEndIncl %d, queryEndIncl %d\n", 
                threadIdx.x, blockIdx.x, alignmentId, subjectLength, 
                positionOfMaxObserved_tileNr,
                positionOfMaxObserved_itemIndex,
                positionOfMaxObserved_y, int(maxObserved),
                score,
                subjectEndIncl,
                queryEndIncl
                );
        }

        if(group.thread_rank() == 0){
            scoreOutput[alignmentId] = score;
            queryEndPositions_inclusive[alignmentId] = queryEndIncl;
            subjectEndPositions_inclusive[alignmentId] = subjectEndIncl;
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
    int currentLetter,
    int row, int tileNr, ScoreType
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
    scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
    E = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(E, gapextendscore));
    scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0],gapextendscore)); //this computes F of the next row !
    scoreDiag = tempM;
    #ifdef USE_E_PRINTARRAY
    Eprintarray[0] = E;
    #endif

    #pragma unroll
    for(int k = 1; k < 4; k++){
        tempM = scoresM[k];
        scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
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
            scoresM[index] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[index]));
            E = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(E, gapextendscore));
            scoresF[index] = MathOps::add_max(scoresM[index], gapopenscore, MathOps::add(scoresF[index], gapextendscore)); //this computes F of the next row !
            scoreDiag = tempM;
            #ifdef USE_E_PRINTARRAY
            Eprintarray[index] = E;
            #endif
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
    __grid_constant__ int* __restrict__ const queryStartPositions_inclusive,
    __grid_constant__ int* __restrict__ const subjectStartPositions_inclusive,
    __grid_constant__ const int* __restrict__ const scores,
    __grid_constant__ const int* __restrict__ const queryEndPositions_inclusive,
    __grid_constant__ const int* __restrict__ const subjectEndPositions_inclusive,
    __grid_constant__ const InputData inputData,
    __grid_constant__ const SUBMAT* __restrict__ const substmatPtr,
    __grid_constant__ const ScoringKernelParam<ScoreType> scoring,
    __grid_constant__ char* __restrict__ const tempStorage,
    __grid_constant__ const size_t tempBytesPerGroup
){
    static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);
    static_assert(penaltyType == PenaltyType::Affine);

    static_assert(groupsize >= 4);
    static_assert(groupsize <= 32);
    static_assert(blocksize % groupsize == 0);

    constexpr int expectedNumColumnsSUBMAT = alphabetSize+1;
    constexpr int expectedNumRowsSUBMAT = alphabetSize+1;

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
    //     LocalAlignmentLinearGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>,
    //     LocalAlignmentAffineGapState_floatOrInt<ScoreType, numItems, decltype(group), SubstitutionScoreProvider, UpdateMaxOp, relaxChunkSize>
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
        assert(subjectLength > 0);
        // assert(subjectLength <= 17);            
			// if(&groupTempStorage[subjectLength + group.thread_rank()] >= groupTempStorageEnd){
                // const auto startEndPositions = devAlignmentScores[alignmentId].getExtra();
                // const int reverseSubjectLength = startEndPositions.getSubjectEndInclusive() + 1;
                // const int reverseQueryLength = startEndPositions.getQueryEndInclusive() + 1;

                // printf("error clear. gtid %d, tid %d, block %d, iteration %d, subjectLength %d, "
                //     "tempBytesPerGroup %lu, groupTempStorage %p, groupTempStorageEnd %p, write to %p"
                //     "reverseSubjectLength %d, reverseQueryLength %d\n",
                //     group.thread_rank(),
                //     threadIdx.x,
                //     blockIdx.x,
                //     iteration,
                //     subjectLength,
                //     tempBytesPerGroup,
                //     groupTempStorage,
                //     groupTempStorageEnd,
                //     &groupTempStorage[subjectLength + group.thread_rank()],
                //     reverseSubjectLength,
                //     reverseQueryLength
                // );
            // }
            groupTempStorage[subjectLength + group.thread_rank()] = TempStorageDataType{};
    };

    for(int alignmentId = groupIdInGrid; alignmentId < inputData.getNumAlignments(); alignmentId += numGroupsInGrid){
        const auto* query = inputData.getQuery(alignmentId);
        int numTiles = 0;

        int queryEndIncl = 0;
        int subjectEndIncl = 0;
        ScoreType scoresF[numItems]{};
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        ScoreType E;
        #ifdef USE_E_PRINTARRAY
        ScoreType Eprintarray[numItems];
        #endif
        ScoreType maxObserved = oobscore;
        int positionOfScoreToFind_y = -1;
        int positionOfScoreToFind_itemIndex = -1;
        int positionOfScoreToFind_tileNr = -1;
        bool groupFoundScore = false;

        int reverseSubjectLength = 0;
        int reverseQueryLength = 0;

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
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            scoreDiag = upTempScore;
            #ifdef USE_E_PRINTARRAY
            Eprintarray[0] = E;
            #endif

            #pragma unroll
            for(int k = 1; k < 4; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
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
                    scoresM[index] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[index]));
                    scoreDiag = upTempScore;
                    #ifdef USE_E_PRINTARRAY
                    Eprintarray[index] = E;
                    #endif
                }
            }

            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }
        };

        auto searchScores = [&](int row, int tileNr, ScoreType scoreToFind){
            if(ifprint()){    
                printState(row);
            }
            int indexOfScoreToFind = -1;
            const int myRow = row - group.thread_rank();
            if(0 < myRow && myRow <= reverseSubjectLength){
                #pragma unroll
                for(int k = 0; k < numItems; k++){
                    // const int dpColumnIndex = tileNr * tileSize + group.thread_rank() * numItems+k;
                    // if(ifprint()){
                    //     printf("tid %d, k %d, dpColumnIndex %d, reverseQueryLength %d, scoresM[k] %3.f, scoreToFind %3.f, comp %d\n", 
                    //         threadIdx.x + blockIdx.x * blockDim.x,
                    //         k,
                    //         dpColumnIndex,
                    //         reverseQueryLength,
                    //         scoresM[k],
                    //         scoreToFind,
                    //         scoreToFind == scoresM[k]
                    //     );
                    // }
                    // if(dpColumnIndex < reverseQueryLength){
                    const int tileColumnIndex = group.thread_rank() * numItems + k;
                    const int globalColumnIndex = tileNr * tileSize + tileColumnIndex;

                    if(globalColumnIndex < reverseQueryLength){
                        if(scoreToFind == scoresM[k]){
                            indexOfScoreToFind = k;
                            // if(ifprint()){
                            //     printf("found. thread %d, row %d, k %d, scoresM[k] %f\n", 
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
                positionOfScoreToFind_tileNr = tileNr;
                positionOfScoreToFind_itemIndex = indexOfScoreToFind;
                positionOfScoreToFind_y = row;
            }
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
            queryEndIncl = queryEndPositions_inclusive[alignmentId];
            subjectEndIncl = subjectEndPositions_inclusive[alignmentId];
            maxObserved = scores[alignmentId];
            subjectData = inputData.getSubject(alignmentId);
            
            reverseSubjectLength = subjectEndIncl + 1;
            reverseQueryLength = queryEndIncl + 1;
            loadOffsetLimit = reverseSubjectLength;
            subjectLoadOffset = 4*group.thread_rank();
            currentLetter = paddingLetter;

            numTiles = SDIV(reverseQueryLength, groupsize * numItems);

            if(ifprint()){
                printf("before reverse. threadid %d, id in group %d, maxObserved %f, endPositionForwardPass(s %d, q %d)\n", 
                    blockIdx.x * blockDim.x + threadIdx.x, group.thread_rank(), 
                    float(maxObserved), 
                    subjectEndIncl, queryEndIncl
                );
            }

            tempWriteOffset = group.thread_rank();

            clearOutOfTileTempStorage(reverseSubjectLength, alignmentId);

            loadQueryLetters_reversed(0, queryLetters); //query
            loadNext4Letters_reversed(); //subject
            initScoresFirstTile();

            const int numRows = (reverseSubjectLength + 1) + (groupsize-1);
            int r = 1;

            const ScoreType& scoreToFind = maxObserved;

            // auto processUntilScoreMatch_withEarlyExit = [&](const ScoreType& scoreToFind){
            #ifdef BACKWARD_EARLY_EXIT_BY_ROW
            do{
                //process first groupsize - 1 diagonals which contain out-of-bound threads

                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                searchScores(r, tileNr, scoreToFind);
                groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                if(groupFoundScore) break;
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //y
                searchScores(r+1, tileNr, scoreToFind);
                groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                if(groupFoundScore) break;
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //z
                searchScores(r+2, tileNr, scoreToFind);
                groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                if(groupFoundScore) break;
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                shuffle4Letters();

                r = 4;
                for(; r < groupsize - 1; r += 4){                    
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                    searchScores(r+3, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();
                }
                //break after loop
                if(groupFoundScore) break;

                //process remaining diagonals. process in chunks of 4 diagonals.
                //for those diagonals we need to store the last column of the tile to temp memory
                //last column is stored in "rightBorder"

                //r starts with r=max(4, groupsize)
                for(; r < numRows - 3; r += 4){

                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z 
                    searchScores(r+3, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                    if((r + 4) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters_reversed();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        // if(&groupTempStorage[tempWriteOffset] >= groupTempStorageEnd){
                        //     printf("error write\n");
                        // }
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    }                    
                }
                //break after loop
                if(groupFoundScore) break;

                //can have at most 3 remaining rows
                if(r < numRows){
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();

                }
                if(r+1 < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores                    
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();
                }
                if(r+2 < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
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
                        // if(&groupTempStorage[tempWriteOffset - firstValidThread] >= groupTempStorageEnd){
                        //     printf("error write\n");
                        // }
                        groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                    }
                }

            }while(false);
            
            #else

            // auto processUntilScoreMatch_noEarlyExit = [&](const ScoreType& scoreToFind){
            {
                //process first groupsize - 1 diagonals which contain out-of-bound threads

                if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                searchScores(r, tileNr, scoreToFind);
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //y
                searchScores(r+1, tileNr, scoreToFind);
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //z
                searchScores(r+2, tileNr, scoreToFind);
                shuffleScoresFirstTile();
                shuffleCurrentLetter();

                if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                shuffle4Letters();

                r = 4;
                for(; r < groupsize - 1; r += 4){                    
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                    searchScores(r+3, tileNr, scoreToFind);
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();
                }

                //process remaining diagonals. process in chunks of 4 diagonals.
                //for those diagonals we need to store the last column of the tile to temp memory
                //last column is stored in "rightBorder"

                //r starts with r=max(4, groupsize)
                for(; r < numRows - 3; r += 4){

                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();


                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z 
                    searchScores(r+3, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter(); 

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                    if((r + 4) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        loadNext4Letters_reversed();
                    }else{
                        //get next 4 letters from neighbor
                        shuffle4Letters();
                    }

                    if((r + 4) % (group.size()) == 0){
                        #ifdef PRINT_WRITE
                        printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                        #endif
                        // if(&groupTempStorage[tempWriteOffset] >= groupTempStorageEnd){
                        //     printf("error write\n");
                        // }
                        groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                        tempWriteOffset += group.size();
                    }                    
                }

                //can have at most 3 remaining rows
                if(r < numRows){
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                    searchScores(r, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();

                }
                if(r+1 < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                    searchScores(r+1, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores                    
                    shuffleScoresFirstTile();
                    shuffleCurrentLetter();
                }
                if(r+2 < numRows){
                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                    searchScores(r+2, tileNr, scoreToFind);
                    shuffleTileLastColumn(); //must be called before setTileLastColumn
                    setTileLastColumn(); //must be called before shuffleScores
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
                        // if(&groupTempStorage[tempWriteOffset - firstValidThread] >= groupTempStorageEnd){
                        //     printf("error write\n");
                        // }
                        groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                    }
                }
            };
            #endif

            // groupFoundScore = processUntilScoreMatch_noEarlyExit(maxObserved);
            groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
        }



        for(int tileNr = 1; tileNr < numTiles; tileNr++){
            int queryLetters[numItems];
            SubstitutionScoreProvider substitutionProvider(shared_substmat, queryLetters);

            /* 
                -----------------------
                Process tile tileNr
                ----------------------- 
            */

            // if(alignmentId < numAlignments && !groupFoundScore && ifprocess()){

            if(needToProcessTile(tileNr)){
            // if(alignmentId < numAlignments){

                subjectLoadOffset = 4*group.thread_rank();
                currentLetter = paddingLetter;
                loadQueryLetters_reversed(tileNr, queryLetters); //query
                loadNext4Letters_reversed(); //subject

                tempWriteOffset = group.thread_rank();

                #ifdef PRINT_LOAD
                printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[group.thread_rank()].x, groupTempStorage[group.thread_rank()].y, group.thread_rank());
                #endif
                // if(&groupTempStorage[group.thread_rank()] >= groupTempStorageEnd){
                //     printf("error load\n");
                // }
                leftBorderM_E = groupTempStorage[group.thread_rank()];
                tempLoadOffset = group.size() + group.thread_rank();


                initScoresNotFirstTile(tileNr);

                const int numRows = (reverseSubjectLength + 1) + (groupsize-1);
                int r = 1;

                const ScoreType& scoreToFind = maxObserved;
                // auto processUntilScoreMatch_withEarlyExit = [&](const ScoreType& scoreToFind){
                #ifdef BACKWARD_EARLY_EXIT_BY_ROW
                do{
                    //process first groupsize - 1 diagonals which contain out-of-bound threads

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                    searchScores(r, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //y
                    searchScores(r+1, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //z
                    searchScores(r+2, tileNr, scoreToFind);
                    groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                    if(groupFoundScore) break;
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();

                    r = 4;
                    for(; r < groupsize - 1; r += 4){                    
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                        searchScores(r+3, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        shuffle4Letters();
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    //process remaining diagonals. process in chunks of 4 diagonals.
                    //for those diagonals we need to store the last column of the tile to temp memory
                    //last column is stored in "rightBorder"

                    //r starts with r=max(4, groupsize)
                    for(; r < numRows - 3; r += 4){

                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            // if(&groupTempStorage[tempLoadOffset] >= groupTempStorageEnd){
                            //     printf("error load\n");
                            // }
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        



                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                        searchScores(r+3, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                        if((r + 4) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            // if(&groupTempStorage[tempWriteOffset] >= groupTempStorageEnd){
                            //     printf("error write\n");
                            // }
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        }
                    }
                    //break after loop
                    if(groupFoundScore) break;

                    //can have at most 3 remaining rows
                    if(r < numRows){
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("last load. tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            // if(&groupTempStorage[tempLoadOffset] >= groupTempStorageEnd){
                            //     printf("error load\n");
                            // }
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter();
                        

                    }
                    if(r+1 < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter();
                        
                    }
                    if(r+2 < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
                        if(groupFoundScore) break;
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                    }

                    const int totalChunksOfFour = reverseSubjectLength / 4;
                    const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                    const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + reverseSubjectLength % 4;
                    if(numThreadsWithValidTileLastColumn > 0){
                        const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                        if(group.thread_rank() >= firstValidThread){
                            #ifdef PRINT_WRITE
                            printf("last write. tid %d, write %f %f\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y);
                            #endif
                            // if(&groupTempStorage[tempWriteOffset - firstValidThread] >= groupTempStorageEnd){
                            //     printf("error write\n");
                            // }
                            groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                        }
                    }

                }while(false);
                
                #else

                // auto processUntilScoreMatch_noEarlyExit = [&](const ScoreType& scoreToFind)
                {
                    //process first groupsize - 1 diagonals which contain out-of-bound threads

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                    relaxFirstDiagonal_backwards_untilScoreMatch(r, tileNr, scoreToFind, substitutionProvider); //x
                    searchScores(r, tileNr, scoreToFind);
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //y
                    searchScores(r+1, tileNr, scoreToFind);
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                    relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //z
                    searchScores(r+2, tileNr, scoreToFind);
                    shuffleLeftBorder(); //must be called before shuffleScores
                    shuffleScoresNotFirstTile();
                    shuffleCurrentLetter();

                    if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                    shuffle4Letters();

                    r = 4;
                    for(; r < groupsize - 1; r += 4){                    
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                        searchScores(r+3, tileNr, scoreToFind);
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }
                        shuffle4Letters();
                    }

                    //process remaining diagonals. process in chunks of 4 diagonals.
                    //for those diagonals we need to store the last column of the tile to temp memory
                    //last column is stored in "rightBorder"

                    //r starts with r=max(4, groupsize)
                    for(; r < numRows - 3; r += 4){

                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            // if(&groupTempStorage[tempLoadOffset] >= groupTempStorageEnd){
                            //     printf("error load\n");
                            // }
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        



                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 
                        

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.z; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+3, tileNr, scoreToFind); //z
                        searchScores(r+3, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter(); 

                        if(group.thread_rank() == 0){ currentLetter = current4Letters.w; }

                        if((r + 4) % (4*group.size()) == 0){
                            //used up all query letters stored across the group. reload
                            loadNext4Letters_reversed();
                        }else{
                            //get next 4 letters from neighbor
                            shuffle4Letters();
                        }

                        if((r + 4) % (group.size()) == 0){
                            #ifdef PRINT_WRITE
                            printf("tid %d, write %f %f to %d\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y, tempWriteOffset);
                            #endif
                            // if(&groupTempStorage[tempWriteOffset] >= groupTempStorageEnd){
                            //     printf("error write\n");
                            // }
                            groupTempStorage[tempWriteOffset] = tileLastColumnM_E;
                            tempWriteOffset += group.size();
                        }
                    }

                    //can have at most 3 remaining rows
                    if(r < numRows){
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r, tileNr, scoreToFind); //w
                        searchScores(r, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        if(r % group.size() == 0 && r < reverseSubjectLength){
                            #ifdef PRINT_LOAD
                            printf("last load. tid %d, load %f %f from %d\n", group.thread_rank(), groupTempStorage[tempLoadOffset].x, groupTempStorage[tempLoadOffset].y, tempLoadOffset);
                            #endif
                            // if(&groupTempStorage[tempLoadOffset] >= groupTempStorageEnd){
                            //     printf("error load\n");
                            // }
                            leftBorderM_E = groupTempStorage[tempLoadOffset];
                            tempLoadOffset += group.size();
                        }else{
                            shuffleLeftBorder(); //must be called before shuffleScores
                        }
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter();
                        

                    }
                    if(r+1 < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.x; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+1, tileNr, scoreToFind); //x
                        searchScores(r+1, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                        shuffleLeftBorder(); //must be called before shuffleScores
                        shuffleScoresNotFirstTile();
                        shuffleCurrentLetter();
                        
                    }
                    if(r+2 < numRows){
                        if(group.thread_rank() == 0){ currentLetter = current4Letters.y; }
                        relax_backwards_untilScoreMatch_freeFunction<ScoreType, tileSize>(group, substitutionProvider, reverseQueryLength, reverseQueryLength, oobscore, gapopenscore, gapextendscore, scoresM, scoresF, E, scoreDiag, currentLetter, r+2, tileNr, scoreToFind); //y
                        searchScores(r+2, tileNr, scoreToFind);
                        shuffleTileLastColumn(); //must be called before setTileLastColumn
                        setTileLastColumn(); //must be called before shuffleScores
                    }

                    const int totalChunksOfFour = reverseSubjectLength / 4;
                    const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
                    const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + reverseSubjectLength % 4;
                    if(numThreadsWithValidTileLastColumn > 0){
                        const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
                        if(group.thread_rank() >= firstValidThread){
                            #ifdef PRINT_WRITE
                            printf("last write. tid %d, write %f %f\n", group.thread_rank(), tileLastColumnM_E.x, tileLastColumnM_E.y);
                            #endif
                            // if(&groupTempStorage[tempWriteOffset - firstValidThread] >= groupTempStorageEnd){
                            //     printf("error write\n");
                            // }
                            groupTempStorage[tempWriteOffset - firstValidThread] = tileLastColumnM_E;
                        }
                    }
                };
                #endif

                groupFoundScore = group.any(positionOfScoreToFind_tileNr >= 0);
            }
        }
        //printState(r+3);

        if(!groupFoundScore){
            printf("tid %d did not find score. alignmentId %d\n", 
                threadIdx.x + blockIdx.x * blockDim.x, alignmentId);

            if(threadIdx.x % groupsize == 0){
                printf("subject\n");
                for(int i = 0; i < reverseSubjectLength; i++){
                    std::int8_t b = subjectData[i];
                    if(b == 0) printf("A");
                    else if(b == 1) printf("C");
                    else if(b == 2) printf("G");
                    else if(b == 3) printf("T");
                    else printf("N");
                }
                printf("\n");
                printf("query\n");
                for(int i = 0; i < reverseQueryLength; i++){
                    std::int8_t b = query[i];
                    if(b == 0) printf("A");
                    else if(b == 1) printf("C");
                    else if(b == 2) printf("G");
                    else if(b == 3) printf("T");
                    else printf("N");
                }
                printf("\n");
            }
        }
        // if(alignmentId == 92800){
        // if(!groupFoundScore){
        //     printf("iteration %d groupFoundScore %d. global thread id %d, blockId %d, id in block %d, id in group %d, maxObserved %f, endPositionForwardPass(s %d, q %d)\n", 
        //         iteration, groupFoundScore, 
        //         blockIdx.x * blockDim.x + threadIdx.x, 
        //         blockIdx.x,
        //         threadIdx.x,
        //         group.thread_rank(), 
        //         float(maxObserved), 
        //         startEndPositions.getSubjectEndInclusive(), startEndPositions.getQueryEndInclusive()
        //     );
        // }

        const int mySubjectBeginTmp = positionOfScoreToFind_y - group.thread_rank() - 1;
        const int myQueryBeginTmp = positionOfScoreToFind_tileNr * tileSize + group.thread_rank() * numItems + positionOfScoreToFind_itemIndex;
        const int3 packed = make_int3(positionOfScoreToFind_tileNr >= 0, 
            myQueryBeginTmp,
            mySubjectBeginTmp);
        const int3 maxPacked = cooperative_groups::reduce(group, packed, [](int3 l, int3 r){
            //found
            if(l.x > r.x) return l;
            if(l.x < r.x) return r;
            //prefer smaller queryBegin
            if(l.y < r.y){
                return l;
            }else{
                return r;
            }

            // if(l.y > r.y) return l;
            // if(l.y < r.y) return r;
            // if(l.z > r.z){
            //     return l;
            // }else{
            //     return r;
            // }
        });

        if(ifprint()){
            printf("tid %d, reduction input. x %d, y %d, z %d.\n",
                threadIdx.x,
                positionOfScoreToFind_tileNr >= 0,
                myQueryBeginTmp,
                mySubjectBeginTmp
            );

            printf("tid %d, reduction output. x %d, y %d, z %d.\n",
                threadIdx.x,
                maxPacked.x,
                maxPacked.y,
                maxPacked.z
            );
        }

        if(group.thread_rank() == 0){
            queryStartPositions_inclusive[alignmentId] = reverseQueryLength - maxPacked.y - 1;
            subjectStartPositions_inclusive[alignmentId] = reverseSubjectLength - maxPacked.z - 1;
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






} //namespace localalignment



#endif