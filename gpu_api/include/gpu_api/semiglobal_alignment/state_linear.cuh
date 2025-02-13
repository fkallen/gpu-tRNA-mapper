#ifndef SEMIGLOBAL_ALIGNMENT_STATE_LINEAR_CUH
#define SEMIGLOBAL_ALIGNMENT_STATE_LINEAR_CUH

#include "../util.cuh"
#include "../mathops.cuh"
#include "state_common.cuh"

#include <cstdio>

namespace semiglobalalignment{


    // #define PRINT_STATE_SEMIGLOBAL_LINEAR

    template<
        class ScoreType_, 
        int numItems, 
        class Group, 
        class SubstitutionScoreProvider,
        class UpdateMaxInLastColumnOp,
        int relaxChunkSize = 4
    >
    struct SemiglobalAlignmentLinearGapState_floatOrInt{
        using ScoreType = ScoreType_;
        static_assert(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>);

        using MathOps = MathOps<ScoreType>;

        static_assert(relaxChunkSize == 4 || relaxChunkSize == 8);
        static_assert(numItems % relaxChunkSize == 0);
    
        const ScoreType gapscore;
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        Group& group;
        const SubstitutionScoreProvider& substitutionScores;
        UpdateMaxInLastColumnOp& updateMaxInLastCol;
    
        __device__
        SemiglobalAlignmentLinearGapState_floatOrInt(const SubstitutionScoreProvider& sub, Group& g, UpdateMaxInLastColumnOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : gapscore(scoring.gapscore), group(g), substitutionScores(sub), updateMaxInLastCol(update){}
    
        template<class LeftBorder>
        __device__
        void initScores(int tileNr, const LeftBorder& leftBorder){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = ScoreType{};
                }
                scoreDiag = ScoreType{};
                updateFromLeftBorder(1, leftBorder);
            }else{
                static constexpr ScoreType oobscore = OOBScore<ScoreType>::get();

                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? ScoreType{} : OOBScore<ScoreType>::get();
            }
        }

        template<class LeftBorder>
        __device__
        void updateFromLeftBorder(int row, const LeftBorder& leftBorder){
            scoreLeft = leftBorder.getM(row, gapscore);
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
                printf("\n");
            }


        }

        template<bool isFirstDiagonal>
        __device__
        void relax(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            const ScoreType upTempScore = scoresM[0];
            const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[0]);
            const ScoreType tmp = MathOps::max(scoreLeft, upTempScore);

            // if constexpr(isFirstDiagonal){
            //     if(row - group.thread_rank() == 0){
            //         //this ensures that scoresM of the 0-th row will be properly computed as gop + (i-1) * gex if more than 1 thread is used.
            //         scoresM[0] = 0;
            //     }else{
            //         scoresM[0] = MathOps::add_max(tmp, gapscore, newdiag);
            //     }
            // }else{
            //     scoresM[0] = MathOps::add_max(tmp, gapscore, newdiag);
            // }
            scoresM[0] = MathOps::add_max(tmp, gapscore, newdiag);
            updateMaxInLastCol(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                const ScoreType upTempScore = scoresM[k];
                const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                const ScoreType tmp = MathOps::max(scoresM[k-1], upTempScore);
                scoresM[k] = MathOps::add_max(tmp, gapscore, newdiag);
                // updateMaxInLastCol(scoresM[k], tileNr, row, k);
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
                    const ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                    const ScoreType tmp = MathOps::max(scoresM[relaxChunkSize*i + k-1], upTempScore);
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max(tmp, gapscore, newdiag);
                    // updateMaxInLastCol(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }

            //initialization of 0-th row in dp matrix for thread rank > 0
            if(row - group.thread_rank() == 0){
                #pragma unroll
                for(int i = 0; i < numItems; i++){
                    scoresM[i] = ScoreType{};
                }
            }

            updateMaxInLastCol(scoresM, tileNr, row);
        };
    
        __device__
        void shuffleScores(ScoreType leftBorderM){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM;
            }else{
                scoreLeft = newscoreLeft;
            }
        }
    
        __device__ __forceinline__
        void stepSingleTileOtherDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("after\n");
                }
                group.sync();
                printState();
            }
            #endif
        }

        __device__ __forceinline__
        void stepSingleTileFirstDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
        class ScoreType_, 
        int numItems, 
        class Group, 
        class SubstitutionScoreProvider,
        class UpdateMaxInLastColumnOp,
        int relaxChunkSize = 4
    >
    struct SemiglobalAlignmentLinearGapState_half2OrShort2{
        using ScoreType = ScoreType_;
        static_assert(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>);

        using MathOps = MathOps<ScoreType>;

        static_assert(relaxChunkSize == 4 || relaxChunkSize == 8);
        static_assert(numItems % relaxChunkSize == 0);

        ScoreType gapscore;
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        Group& group;
        const SubstitutionScoreProvider& substitutionScores;
        UpdateMaxInLastColumnOp& updateMaxInLastCol;
    
        __device__
        SemiglobalAlignmentLinearGapState_half2OrShort2(const SubstitutionScoreProvider& sub, Group& g, UpdateMaxInLastColumnOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : 
            // gapscore(scoring.gapscore), 
            group(g), 
            substitutionScores(sub), 
            updateMaxInLastCol(update){

            /*
                Compiler (12.6) does not like direct initialization of gapscore
                with short2 and will insert many prmt instructions during computations
                Explicit unpacking and re-packing prevents that
            */
            const auto temp1 = scoring.gapscore.x;
            const ScoreType temp11 = make_vec2<ScoreType>(temp1, temp1);
            gapscore = temp11;

        }
    
        template<class LeftBorder>
        __device__
        void initScores(int tileNr, const LeftBorder& leftBorder){
            if(group.thread_rank() == 0){
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = ScoreType{};
                }
                scoreDiag = ScoreType{};
                updateFromLeftBorder(1, leftBorder);
            }else{
                const ScoreType oobscore = OOBScore<ScoreType>::get();
                #pragma unroll
                for (int i=0; i < numItems; i++) {
                    scoresM[i] = oobscore;
                }
                scoreDiag = oobscore;
                scoreLeft = group.thread_rank() == 1 ? ScoreType{} : OOBScore<ScoreType>::get();;
            }
        }

        template<class LeftBorder>
        __device__
        void updateFromLeftBorder(int row, const LeftBorder& leftBorder){
            scoreLeft = leftBorder.getM(row, gapscore);
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
                    printf("scoreLeft: (%3f %3f), scoreDiag (%3f %3f)\n", 
                        float(scoreLeft.x), float(scoreLeft.y), float(scoreDiag.x), float(scoreDiag.y));
                }
                group.sync();
            }
            if(group.thread_rank() == 0){
                printf("\n");
            }


        }

        template<bool isFirstDiagonal>
        __device__
        void relax(int currentLetter, int row, int tileNr){
            ScoreType fooArray[relaxChunkSize];
            if constexpr(relaxChunkSize == 4){
                substitutionScores.loadFour(group, fooArray, currentLetter, 0);
            }else if(relaxChunkSize == 8){
                substitutionScores.loadEight(group, fooArray, currentLetter, 0);
            }

            const ScoreType upTempScore = scoresM[0];
            const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[0]);
            const ScoreType tmp = MathOps::max(scoreLeft, upTempScore);
            scoresM[0] = MathOps::add_max(tmp, gapscore, newdiag);
            updateMaxInLastCol(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                const ScoreType upTempScore = scoresM[k];
                const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                const ScoreType tmp = MathOps::max(scoresM[k-1], upTempScore);
                scoresM[k] = MathOps::add_max(tmp, gapscore, newdiag);
                updateMaxInLastCol(scoresM[k], tileNr, row, k);
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
                    const ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                    const ScoreType tmp = MathOps::max(scoresM[relaxChunkSize*i + k-1], upTempScore);
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max(tmp, gapscore, newdiag);
                    updateMaxInLastCol(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
                }
            }

            //initialization of 0-th row in dp matrix for thread rank > 0
            if(row - group.thread_rank() == 0){
                #pragma unroll
                for(int i = 0; i < numItems; i++){
                    scoresM[i] = ScoreType{};
                }
            }
        };
    
        __device__
        void shuffleScores(ScoreType leftBorderM){
            scoreDiag = scoreLeft;
            const ScoreType newscoreLeft = group.shfl_up(scoresM[numItems-1], 1);
            if(group.thread_rank() == 0){
                scoreLeft = leftBorderM;
            }else{
                scoreLeft = newscoreLeft;
            }
        }
    
        __device__ __forceinline__
        void stepSingleTileOtherDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("after\n");
                }
                group.sync();
                printState();
            }
            #endif
        }

        __device__ __forceinline__
        void stepSingleTileFirstDiagonal(int subject_letter, int row){
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = false;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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
            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
                printf("tid %d, subject_letter %d, row %d\n", threadIdx.x, subject_letter, row);
            }
            if(group.meta_group_rank() == 0){
                if(threadIdx.x == 0){
                    printf("before\n");
                }
                group.sync();
                printState();
            }
            #endif

            constexpr bool isFirstDiagonal = true;
            relax<isFirstDiagonal>(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_SEMIGLOBAL_LINEAR
            if(group.meta_group_rank() == 0){
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