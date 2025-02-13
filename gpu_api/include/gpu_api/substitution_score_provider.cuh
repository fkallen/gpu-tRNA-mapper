#ifndef SUBSTITUTION_SCORE_PROVIDER
#define SUBSTITUTION_SCORE_PROVIDER

#include "util.cuh"

#include <cuda/std/array>

    template<class SharedSubstitutionMatrix, class ScoreType, int numItems>
    struct SubstitutionMatrixSubstitutionScoreProvider{

        const int (&queryLetters)[numItems];
        const SharedSubstitutionMatrix& substitutionMatrix;

        __device__
        SubstitutionMatrixSubstitutionScoreProvider(const SharedSubstitutionMatrix& shared_substmat_, const int (&queryLetters_)[numItems]) 
            : substitutionMatrix(shared_substmat_), queryLetters(queryLetters_){}

        template<class Group>
        __device__
        void loadFour(Group&, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            #pragma unroll
            for(int k = 0; k < 4; k++){
                outputArray[k] = substitutionMatrix.data[subjectLetter][queryLetters[4*ithGroupwideLoad + k]];
            }
        }
    };

    template<
        class SharedSubstitutionMatrix, 
        class ScoreType, 
        int numItems,
        int alphabetSize,
        bool doPackSubject,
        bool doPackQuery
    >
    struct SubstitutionMatrixPackingSubstitutionScoreProvider{
        static_assert(doPackSubject || doPackQuery);
        
        const int (&queryLetters)[numItems];
        const SharedSubstitutionMatrix& substitutionMatrix;

        __device__
        SubstitutionMatrixPackingSubstitutionScoreProvider(const SharedSubstitutionMatrix& shared_substmat_, const int (&queryLetters_)[numItems]) 
            : substitutionMatrix(shared_substmat_), queryLetters(queryLetters_){}

        template<class Group>
        __device__
        void loadFour(Group&, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            if constexpr(std::is_same_v<ScoreType, typename SharedSubstitutionMatrix::value_type>){
                static_assert(doPackSubject != doPackQuery);
                if constexpr(doPackSubject){
                    //matrix has layout [dim][dim*dim] with vector score type, i.e. half2
                    const int subjectLetter0 = (subjectLetter / alphabetSize);
                    const int subjectLetter1 = (subjectLetter % alphabetSize);
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        outputArray[k] = make_vec2<ScoreType>(
                            substitutionMatrix.data[subjectLetter0][queryLetters[4*ithGroupwideLoad + k]].x,
                            substitutionMatrix.data[subjectLetter1][queryLetters[4*ithGroupwideLoad + k]].y
                        );
                    }
                }else{
                    static_assert(doPackQuery);
                    //matrix has layout [dim*dim][dim] with vector score type, i.e. half2
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        const int queryLetter0 = (queryLetters[4*ithGroupwideLoad + k] / alphabetSize);
                        const int queryLetter1 = (queryLetters[4*ithGroupwideLoad + k] % alphabetSize);
                        outputArray[k] = make_vec2<ScoreType>(
                            substitutionMatrix.data[subjectLetter][queryLetter0].x,
                            substitutionMatrix.data[subjectLetter][queryLetter1].y
                        );
                    }
                }
            }else{
                if constexpr (doPackSubject && doPackQuery){
                    const int subjectLetter0 = (subjectLetter / alphabetSize);
                    const int subjectLetter1 = (subjectLetter % alphabetSize);
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        const int queryLetter0 = (queryLetters[4*ithGroupwideLoad + k] / alphabetSize);
                        const int queryLetter1 = (queryLetters[4*ithGroupwideLoad + k] % alphabetSize);
                        outputArray[k] = make_vec2<ScoreType>(
                            substitutionMatrix.data[subjectLetter0][queryLetter0],
                            substitutionMatrix.data[subjectLetter1][queryLetter1]
                        );
                    }
                }else if(doPackSubject){
                    const int subjectLetter0 = (subjectLetter / alphabetSize);
                    const int subjectLetter1 = (subjectLetter % alphabetSize);
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        outputArray[k] = make_vec2<ScoreType>(
                            substitutionMatrix.data[subjectLetter0][queryLetters[4*ithGroupwideLoad + k]],
                            substitutionMatrix.data[subjectLetter1][queryLetters[4*ithGroupwideLoad + k]]
                        );
                    }
                }else if(doPackQuery){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        const int queryLetter0 = (queryLetters[4*ithGroupwideLoad + k] / alphabetSize);
                        const int queryLetter1 = (queryLetters[4*ithGroupwideLoad + k] % alphabetSize);
                        outputArray[k] = make_vec2<ScoreType>(
                            substitutionMatrix.data[subjectLetter][queryLetter0],
                            substitutionMatrix.data[subjectLetter][queryLetter1]
                        );
                    }
                }
                
            }
        }
    };

    template<class ScoreType, int numItems, int alphabetSize>
    struct MatchMismatchSubstitutionScoreProvider{

        const int (&queryLetters)[numItems];
        const ScoreType matchscore;
        const ScoreType mismatchscore;

        __device__
        MatchMismatchSubstitutionScoreProvider(const int (&queryLetters_)[numItems], ScoreType matchscore_, ScoreType mismatchscore_) 
            : queryLetters(queryLetters_), matchscore(matchscore_), mismatchscore(mismatchscore_){}

        template<class Group>
        __device__
        void loadFourFloatOrInt(Group&, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            #pragma unroll
            for(int k = 0; k < 4; k++){
                outputArray[k] = (subjectLetter == queryLetters[4*ithGroupwideLoad + k]) ? matchscore : mismatchscore;
            }
        }

        template<class Group>
        __device__
        void loadFourHalf2OrShort2(Group&, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            #pragma unroll
            for(int k = 0; k < 4; k++){
                outputArray[k].x = ((subjectLetter / alphabetSize) == queryLetters[4*ithGroupwideLoad + k]) ? matchscore.x : mismatchscore.x;
                outputArray[k].y = ((subjectLetter % alphabetSize) == queryLetters[4*ithGroupwideLoad + k]) ? matchscore.y : mismatchscore.y;
            }
        }

        template<class Group>
        __device__
        void loadFour(Group& group, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            if constexpr(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>){
                loadFourFloatOrInt(group, outputArray, subjectLetter, ithGroupwideLoad);
            }else if(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>){
                loadFourHalf2OrShort2(group, outputArray, subjectLetter, ithGroupwideLoad);
            }
        }


    };


    /*
        to process two independent subjects / queries
        compare subject.x to query.x and subject.y to query.y
    */
    template<class ScoreType, int numItems, int alphabetSize>
    struct TwoIndependentMatchMismatchSubstitutionScoreProvider{

        const int (&queryLetters)[numItems];
        const ScoreType matchscore;
        const ScoreType mismatchscore;

        __device__
        TwoIndependentMatchMismatchSubstitutionScoreProvider(const int (&queryLetters_)[numItems], ScoreType matchscore_, ScoreType mismatchscore_) 
            : queryLetters(queryLetters_), matchscore(matchscore_), mismatchscore(mismatchscore_){}

        template<class Group>
        __device__
        void loadFourHalf2OrShort2(Group&, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            #pragma unroll
            for(int k = 0; k < 4; k++){
                outputArray[k].x = ((subjectLetter / alphabetSize) == (queryLetters[4*ithGroupwideLoad + k] / alphabetSize)) ? matchscore.x : mismatchscore.x;
                outputArray[k].y = ((subjectLetter % alphabetSize) == (queryLetters[4*ithGroupwideLoad + k] % alphabetSize)) ? matchscore.y : mismatchscore.y;
            }
        }

        template<class Group>
        __device__
        void loadFour(Group& group, ScoreType (&outputArray)[4], int subjectLetter, int ithGroupwideLoad) const{
            loadFourHalf2OrShort2(group, outputArray, subjectLetter, ithGroupwideLoad);
        }


    };










#endif