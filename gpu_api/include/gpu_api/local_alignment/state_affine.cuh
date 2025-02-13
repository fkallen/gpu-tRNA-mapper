#ifndef LOCAL_ALIGNMENT_STATE_AFFINE_CUH
#define LOCAL_ALIGNMENT_STATE_AFFINE_CUH

#include "../util.cuh"
#include "../mathops.cuh"
#include "state_common.cuh"

#include <cstdio>

namespace localalignment{

    // #define PRINT_STATE_LOCAL_AFFINE
    // #define PRINT_STATE_LOCAL_AFFINE_HALF2

    #define USE_SIX_MAX_FOUR_ADD_VERSION

    template<
        class ScoreType, 
        int numItems, 
        class Group, 
        class SubstitutionScoreProvider,
        class UpdateMaxOp,
        int relaxChunkSize = 4
    >
    struct LocalAlignmentAffineGapState_floatOrInt{
        static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

        using MathOps = MathOps<ScoreType>;

        static_assert(relaxChunkSize == 4 || relaxChunkSize == 8);
        static_assert(numItems % relaxChunkSize == 0);
    
        const ScoreType gapopenscore;
        const ScoreType gapextendscore;
        ScoreType scoresF[numItems]{};
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        ScoreType E;
        Group& group;
        const SubstitutionScoreProvider& substitutionScores;
        UpdateMaxOp& updateMax;
    
        __device__
        LocalAlignmentAffineGapState_floatOrInt(const SubstitutionScoreProvider& sub, Group& g, UpdateMaxOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : gapopenscore(scoring.gapopenscore), gapextendscore(scoring.gapextendscore), group(g), substitutionScores(sub), updateMax(update){}
    
        template<class LeftBorder>
        __device__
        void initScores(int tileNr, const LeftBorder& leftBorder){
            static constexpr ScoreType oobscore = OOBScore<ScoreType>::get();

            if(group.thread_rank() == 0){
                
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = ScoreType{};
                    scoresF[i] = oobscore;
                }
                scoreDiag = ScoreType{};
                updateFromLeftBorder(1, leftBorder);
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? ScoreType{} : oobscore;
                E = oobscore;
            }
        }

        template<class LeftBorder>
        __device__
        void updateFromLeftBorder(int row, const LeftBorder& leftBorder){
            scoreLeft = leftBorder.getM(row, gapopenscore, gapextendscore);
            E = leftBorder.getE();
        }

        __device__
        void printState(){

            if(group.thread_rank() == 0){
                printf("M\n");
            }
            for(int t = 0; t < group.size(); t++){
                if(t == group.thread_rank()){
                    for(int i = 0; i < numItems; i++){
                        printf("%3f ", scoresM[i]);
                    }
                    printf("\n");
                    printf("scoreLeft: %3f, scoreDiag %3f\n", scoreLeft, scoreDiag);
                }
                group.sync();
            }
            if(group.thread_rank() == 0){
                printf("F\n");
            }
            for(int t = 0; t < group.size(); t++){
                if(t == group.thread_rank()){
                    for(int i = 0; i < numItems; i++){
                        printf("%3f ", scoresF[i]);
                    }
                    printf("\n");
                    printf("E: %3f\n", E);
                }
                group.sync();
            }
            if(group.thread_rank() == 0){
                printf("\n");
            }


        };

        #ifdef USE_SIX_MAX_FOUR_ADD_VERSION

        template<bool isFirstTile>
        __device__
        void relaxFirstDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            //in the first tile E is always computed. In succeeding tiles, E is already computed for the first thread (loaded from temp storage)
            if constexpr(isFirstTile){
                E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
            }else{
                if(group.thread_rank() > 0){
                    E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                }
            }

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    E = MathOps::add_max(scoresM[relaxChunkSize*i + k-1], gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresM[relaxChunkSize*i + k], gapopenscore, MathOps::add(scoresF[relaxChunkSize*i + k], gapextendscore));
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }

            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            #ifdef PRINT_STATE_LOCAL_AFFINE
                printf("tid %d, compute advanced E = %f\n", group.thread_rank(), E);
            #endif
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }
        }



        __device__
        void relaxOtherDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            // E and scoresF[0] are already computed

            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;
            ScoreType temp = MathOps::add(scoresM[0], gapopenscore);
            E = MathOps::add_max(E, gapextendscore, temp);
            scoresF[0] = MathOps::add_max(scoresF[0], gapextendscore, temp); //this computes F of the next row !

            #ifdef PRINT_STATE_LOCAL_AFFINE
                printf("tid %d, compute E = %f\n", group.thread_rank(), E);
            #endif

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
                ScoreType temp = MathOps::add(scoresM[k], gapopenscore);
                E = MathOps::add_max(E, gapextendscore, temp);
                scoresF[k] = MathOps::add_max(scoresF[k], gapextendscore, temp); //this computes F of the next row !

                #ifdef PRINT_STATE_LOCAL_AFFINE
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                    ScoreType temp = MathOps::add(scoresM[relaxChunkSize*i + k], gapopenscore);
                    E = MathOps::add_max(E, gapextendscore, temp);
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresF[relaxChunkSize*i + k], gapextendscore, temp); //this computes F of the next row !
                    
                    #ifdef PRINT_STATE_LOCAL_AFFINE
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                }
            }
        }

        #else


        template<bool isFirstTile>
        __device__
        void relaxFirstDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }


            E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
            #ifdef PRINT_STATE_LOCAL_AFFINE
                printf("tid %d, compute E = %f\n", group.thread_rank(), E);
            #endif

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    E = MathOps::add_max(scoresM[relaxChunkSize*i + k-1], gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresM[relaxChunkSize*i + k], gapopenscore, MathOps::add(scoresF[relaxChunkSize*i + k], gapextendscore));
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }
        }



        __device__
        void relaxOtherDiagonal_impl(int currentLetter, int row, int tileNr){
            constexpr bool unusedtemplateparameter = false;
            relaxFirstDiagonal_impl<unusedtemplateparameter>(currentLetter, row, tileNr);
        }

        #endif
    
        __device__
        void shuffleScores(ScoreType leftBorderM, ScoreType leftBorderE){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM;
                E = leftBorderE;
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        }



        __device__ __forceinline__
        void stepSingleTileFirstDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
            
            constexpr bool isFirstTile = true;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, 0);

            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        __device__ __forceinline__
        void stepSingleTileOtherDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
            
            relaxOtherDiagonal_impl(subject_letter, row, 0);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileFirstDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = true;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileOtherDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileFirstDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = false;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileFirstDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = false;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

    };


    template<
        class ScoreType, 
        int numItems, 
        class Group, 
        class SubstitutionScoreProvider,
        class UpdateMaxOp,
        int relaxChunkSize = 4
    >
    struct LocalAlignmentAffineGapState_half2OrShort2{
        static_assert(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>);

        using MathOps = MathOps<ScoreType>;

        static_assert(relaxChunkSize == 4 || relaxChunkSize == 8);
        static_assert(numItems % relaxChunkSize == 0);
    
        ScoreType gapopenscore;
        ScoreType gapextendscore;
        ScoreType scoresF[numItems]{};
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        ScoreType E;
        Group& group;
        const SubstitutionScoreProvider& substitutionScores;
        UpdateMaxOp& updateMax;
    
        __device__
        LocalAlignmentAffineGapState_half2OrShort2(const SubstitutionScoreProvider& sub, Group& g, UpdateMaxOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : 
            // gapopenscore(scoring.gapopenscore), 
            // gapextendscore(scoring.gapextendscore),
            group(g), 
            substitutionScores(sub), 
            updateMax(update){

            /*
                Compiler (12.6) does not like direct initialization of gap open / gap extend
                with short2 and will insert many prmt instructions during computations
                Explicit unpacking and re-packing prevents that
            */
            const auto temp1 = scoring.gapopenscore.x;
            const auto temp2 = scoring.gapextendscore.x;
            const ScoreType temp11 = make_vec2<ScoreType>(temp1, temp1);
            const ScoreType temp22 = make_vec2<ScoreType>(temp2, temp2);
            gapopenscore = temp11;
            gapextendscore = temp22;
        }
    
        template<class LeftBorder>
        __device__
        void initScores(int tileNr, const LeftBorder& leftBorder){
            const ScoreType oobscore = OOBScore<ScoreType>::get();
            if(group.thread_rank() == 0){
                
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = ScoreType{};
                    scoresF[i] = oobscore;
                }
                scoreDiag = ScoreType{};
                updateFromLeftBorder(1, leftBorder);
            }else{
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                    scoresF[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? ScoreType{} : oobscore;
                E = oobscore;
            }
        }

        template<class LeftBorder>
        __device__
        void updateFromLeftBorder(int row, const LeftBorder& leftBorder){
            scoreLeft = leftBorder.getM(row, gapopenscore, gapextendscore);
            E = leftBorder.getE();
        }

        __device__
        void printState(){

            if(group.thread_rank() == 0){
                printf("M\n");
            }
            for(int t = 0; t < group.size(); t++){
                if(t == group.thread_rank()){
                    for(int i = 0; i < numItems; i++){
                        printf("(%3f %3f) ", float(scoresM[i].x), float(scoresM[i].y));
                    }
                    printf("\n");
                    printf("scoreLeft: (%3f %3f), scoreDiag (%3f %3f)\n", float(scoreLeft.x), float(scoreLeft.y), float(scoreDiag.x), float(scoreDiag.y));
                }
                group.sync();
            }
            if(group.thread_rank() == 0){
                printf("F\n");
            }
            for(int t = 0; t < group.size(); t++){
                if(t == group.thread_rank()){
                    for(int i = 0; i < numItems; i++){
                        printf("(%3f %3f) ", float(scoresF[i].x), float(scoresF[i].y));
                    }
                    printf("\n");
                    printf("E: (%3f %3f)\n", float(E.x), float(E.y));
                }
                group.sync();
            }
            if(group.thread_rank() == 0){
                printf("\n");
            }
        }


        #ifdef USE_SIX_MAX_FOUR_ADD_VERSION

        template<bool isFirstTile>
        __device__
        void relaxFirstDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            //in the first tile E is always computed. In succeeding tiles, E is already computed for the first thread (loaded from temp storage)
            if constexpr(isFirstTile){
                E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
            }else{
                if(group.thread_rank() > 0){
                    E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                }
            }

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    E = MathOps::add_max(scoresM[relaxChunkSize*i + k-1], gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresM[relaxChunkSize*i + k], gapopenscore, MathOps::add(scoresF[relaxChunkSize*i + k], gapextendscore));
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }

            //advance E by 1 column and F by 1 row to allow for optimized computations of remaining diagonals
            E = MathOps::add_max(scoresM[numItems-1], gapopenscore, MathOps::add(E, gapextendscore));
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                printf("tid %d, compute advanced E = %f\n", group.thread_rank(), E);
            #endif
            for(int k = 0; k < numItems; k++){
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
            }
        }



        __device__
        void relaxOtherDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            // E and scoresF[0] are already computed

            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;
            ScoreType temp = MathOps::add(scoresM[0], gapopenscore);
            E = MathOps::add_max(E, gapextendscore, temp);
            scoresF[0] = MathOps::add_max(scoresF[0], gapextendscore, temp); //this computes F of the next row !

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                printf("tid %d, compute E = %f\n", group.thread_rank(), E);
            #endif

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
                ScoreType temp = MathOps::add(scoresM[k], gapopenscore);
                E = MathOps::add_max(E, gapextendscore, temp);
                scoresF[k] = MathOps::add_max(scoresF[k], gapextendscore, temp); //this computes F of the next row !

                #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                    ScoreType temp = MathOps::add(scoresM[relaxChunkSize*i + k], gapopenscore);
                    E = MathOps::add_max(E, gapextendscore, temp);
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresF[relaxChunkSize*i + k], gapextendscore, temp); //this computes F of the next row !
                    
                    #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                }
            }
        }

        #else


        template<bool isFirstTile>
        __device__
        void relaxFirstDiagonal_impl(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            E = MathOps::add_max(scoreLeft, gapopenscore, MathOps::add(E, gapextendscore));
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                printf("tid %d, compute E = %f\n", group.thread_rank(), E);
            #endif

            scoresF[0] = MathOps::add_max(scoresM[0], gapopenscore, MathOps::add(scoresF[0], gapextendscore));
            ScoreType upTempScore = scoresM[0];
            scoresM[0] = MathOps::add_max_relu(scoreDiag, fooArray[0], MathOps::max(E, scoresF[0]));
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                E = MathOps::add_max(scoresM[k-1], gapopenscore, MathOps::add(E, gapextendscore));
                #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                    printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                #endif
                scoresF[k] = MathOps::add_max(scoresM[k], gapopenscore, MathOps::add(scoresF[k], gapextendscore));
                ScoreType upTempScore = scoresM[k];
                scoresM[k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[k]));
                updateMax(scoresM[k], tileNr, row, k);
                scoreDiag = upTempScore;
            }

            #pragma unroll
            for(int i = 1; i < numItems/relaxChunkSize; i++){
                if constexpr(relaxChunkSize == 4){
                    substitutionScores.loadFour(group, fooArray, currentLetter, i);
                }else if(relaxChunkSize == 8){
                    substitutionScores.loadEight(group, fooArray, currentLetter, i);
                }

                #pragma unroll
                for(int k = 0; k < relaxChunkSize; k++){
                    E = MathOps::add_max(scoresM[relaxChunkSize*i + k-1], gapopenscore, MathOps::add(E, gapextendscore));
                    #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                        printf("tid %d, compute E = %f\n", group.thread_rank(), E);
                    #endif
                    scoresF[relaxChunkSize*i + k] = MathOps::add_max(scoresM[relaxChunkSize*i + k], gapopenscore, MathOps::add(scoresF[relaxChunkSize*i + k], gapextendscore));
                    ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(scoreDiag, fooArray[k], MathOps::max(E, scoresF[relaxChunkSize*i + k]));
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }
        }



        __device__
        void relaxOtherDiagonal_impl(int currentLetter, int row, int tileNr){
            constexpr bool unusedtemplateparameter = false;
            relaxFirstDiagonal_impl<unusedtemplateparameter>(currentLetter, row, tileNr);
        }


        #endif
    
        __device__
        void shuffleScores(ScoreType leftBorderM, ScoreType leftBorderE){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            const ScoreType newE = group.shfl_up(E, 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM;
                E = leftBorderE;
            }else{
                scoreLeft = newscoreLeft;
                E = newE;
            }
        }



        __device__ __forceinline__
        void stepSingleTileFirstDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
            
            constexpr bool isFirstTile = true;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, 0);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            const ScoreType oobscore = OOBScore<ScoreType>::get();
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        __device__ __forceinline__
        void stepSingleTileOtherDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
            
            relaxOtherDiagonal_impl(subject_letter, row, 0);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            const ScoreType oobscore = OOBScore<ScoreType>::get();
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileFirstDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = true;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            const ScoreType oobscore = OOBScore<ScoreType>::get();
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileOtherDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            #ifdef USE_SIX_MAX_FOUR_ADD_VERSION
            shuffleScores(ScoreType{}, gapopenscore);
            #else
            const ScoreType oobscore = OOBScore<ScoreType>::get();
            shuffleScores(ScoreType{}, oobscore);
            #endif

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileFirstDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = false;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1], E);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }
    
        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileFirstDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            constexpr bool isFirstTile = false;
            relaxFirstDiagonal_impl<isFirstTile>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
                }
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("before\n");
                    }
                    group.sync();
                    printState();
                }
            #endif

            relaxOtherDiagonal_impl(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapopenscore, gapextendscore), leftBorder.getE());

            #ifdef PRINT_STATE_LOCAL_AFFINE_HALF2
                if(blockIdx.x == 0 && group.meta_group_rank() == 0){
                    if(threadIdx.x == 0){
                        printf("after\n");
                    }
                    group.sync();
                    printState();
                }
            #endif
        }

    };






}

#endif