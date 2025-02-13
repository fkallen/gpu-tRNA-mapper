#ifndef LOCAL_ALIGNMENT_STATE_LINEAR_CUH
#define LOCAL_ALIGNMENT_STATE_LINEAR_CUH

#include "../util.cuh"
#include "../mathops.cuh"
#include "state_common.cuh"

#include <cstdio>

namespace localalignment{


    // #define PRINT_STATE_LOCAL_LINEAR

    template<
        class ScoreType, 
        int numItems, 
        class Group, 
        class SubstitutionScoreProvider,
        class UpdateMaxOp,
        int relaxChunkSize = 4
    >
    struct LocalAlignmentLinearGapState_floatOrInt{
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
        UpdateMaxOp& updateMax;
    
        __device__
        LocalAlignmentLinearGapState_floatOrInt(const SubstitutionScoreProvider& s, Group& g, UpdateMaxOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : gapscore(scoring.gapscore), group(g), substitutionScores(s), updateMax(update){}
    
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


        };

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
            scoresM[0] = MathOps::add_max_relu(tmp, gapscore, newdiag);
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                const ScoreType upTempScore = scoresM[k];
                const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                const ScoreType tmp = MathOps::max(scoresM[k-1], upTempScore);
                scoresM[k] = MathOps::add_max_relu(tmp, gapscore, newdiag);
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
                    const ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                    const ScoreType tmp = MathOps::max(scoresM[relaxChunkSize*i + k-1], upTempScore);
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(tmp, gapscore, newdiag);
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
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
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepSingleTileOtherDiagonal(subject_letter, row);
        }
    
        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileOtherDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepFirstTileOtherDiagonal(subject_letter, row, lastColumn);
        }
    
        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepIntermediateTileOtherDiagonal(subject_letter, row, tileNr, leftBorder, lastColumn);
        }
    
        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepLastTileOtherDiagonal(subject_letter, row, tileNr, leftBorder);
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
    struct LocalAlignmentLinearGapState_half2OrShort2{
        using MathOps = MathOps<ScoreType>;

        static_assert(relaxChunkSize == 4 || relaxChunkSize == 8);
        static_assert(numItems % relaxChunkSize == 0);
    
        ScoreType gapscore;
        ScoreType scoresM[numItems]{};
        ScoreType scoreLeft;
        ScoreType scoreDiag;
        Group& group;
        const SubstitutionScoreProvider& substitutionScores;
        UpdateMaxOp& updateMax;
    
        __device__
        LocalAlignmentLinearGapState_half2OrShort2(const SubstitutionScoreProvider& s, Group& g, UpdateMaxOp& update, const ScoringKernelParam<ScoreType>& scoring
        ) : 
            // gapscore(scoring.gapscore), 
            group(g), 
            substitutionScores(s), 
            updateMax(update){

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


        };

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
            scoresM[0] = MathOps::add_max_relu(tmp, gapscore, newdiag);
            updateMax(scoresM[0], tileNr, row, 0);
            scoreDiag = upTempScore;

            #pragma unroll
            for(int k = 1; k < relaxChunkSize; k++){
                const ScoreType upTempScore = scoresM[k];
                const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                const ScoreType tmp = MathOps::max(scoresM[k-1], upTempScore);
                scoresM[k] = MathOps::add_max_relu(tmp, gapscore, newdiag);
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
                    const ScoreType upTempScore = scoresM[relaxChunkSize*i + k];
                    const ScoreType newdiag = MathOps::add(scoreDiag, fooArray[k]);
                    const ScoreType tmp = MathOps::max(scoresM[relaxChunkSize*i + k-1], upTempScore);
                    scoresM[relaxChunkSize*i + k] = MathOps::add_max_relu(tmp, gapscore, newdiag);
                    updateMax(scoresM[relaxChunkSize*i + k], tileNr, row, relaxChunkSize*i + k);
                    scoreDiag = upTempScore;
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
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, 0);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepSingleTileOtherDiagonal(subject_letter, row);
        }
    
        template<class LastColumn>
        __device__ __forceinline__
        void stepFirstTileOtherDiagonal(int subject_letter, int row, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, 0);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(ScoreType{});

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepFirstTileOtherDiagonal(subject_letter, row, lastColumn);
        }
    
        template<class LeftBorder, class LastColumn>
        __device__ __forceinline__
        void stepIntermediateTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder, LastColumn& lastColumn){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, tileNr);
            lastColumn.update(group, row, scoresM[numItems-1]);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepIntermediateTileOtherDiagonal(subject_letter, row, tileNr, leftBorder, lastColumn);
        }
    
        template<class LeftBorder>
        __device__ __forceinline__
        void stepLastTileOtherDiagonal(int subject_letter, int row, int tileNr, LeftBorder& leftBorder){
            #ifdef PRINT_STATE_LOCAL_LINEAR
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

            relax(subject_letter, row, tileNr);
            shuffleScores(leftBorder.getM(row, gapscore));

            #ifdef PRINT_STATE_LOCAL_LINEAR
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
            stepLastTileOtherDiagonal(subject_letter, row, tileNr, leftBorder);
        }
    };



}

#endif