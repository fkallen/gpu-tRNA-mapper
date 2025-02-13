#ifndef SEMIGLOBAL_ALIGNMENT_STATE_COMMON_CUH
#define SEMIGLOBAL_ALIGNMENT_STATE_COMMON_CUH

#include "../util.cuh"
#include "../mathops.cuh"

#include <cstdio>

namespace semiglobalalignment{

    template<class ScoreType>
    struct UpdateMax{
        using MathOps = MathOps<ScoreType>;

        ScoreType maximum{};

        __device__
        void operator()(ScoreType toCompare, int /*tileNr*/, int /*row*/, int /*itemIndex*/){
            maximum = MathOps::max(maximum, toCompare);
        }
    };

    template<class ScoreType, int numItems>
    struct UpdateMaxInLastColumn{
        using MathOps = MathOps<ScoreType>;

        ScoreType maxima[numItems]{};

        __device__
        void operator()(ScoreType toCompare, int /*tileNr*/, int /*row*/, int itemIndex){

            // maxima[itemIndex] = MathOps::max(maxima[itemIndex], toCompare);

            #pragma unroll
            for(int i = 0; i < numItems; i++){
                if(i == itemIndex){
                    maxima[i] = MathOps::max(maxima[i], toCompare);
                }
            }


            // if(threadIdx.x == 0){
                // printf("tid %d, row %d, itemIndex %d, max is %f\n", threadIdx.x, row, itemIndex, maxima[itemIndex]);
            // }
        }

        __device__
        void operator()(const ScoreType (&toCompare)[numItems], int /*tileNr*/, int row){
            #pragma unroll
            for(int i = 0; i < numItems; i++){
                maxima[i] = MathOps::max(maxima[i], toCompare[i]);
            }
        }
    };

    template<class ScoreType>
    struct UpdateMaxInLastColumnSingleReg{
        using MathOps = MathOps<ScoreType>;

        int selectedItemIndex;
        ScoreType maxima{};

        __device__
        UpdateMaxInLastColumnSingleReg(int which) : selectedItemIndex(which) {
        }

        __device__
        void operator()(ScoreType toCompare, int /*tileNr*/, int row, int itemIndex){
            if(itemIndex == selectedItemIndex){
                maxima = MathOps::max(maxima, toCompare);
            }
        }
    };

    template<class ScoreType, int numItems>
    struct UpdateMaxInLastColumnWithPointer{
        using MathOps = MathOps<ScoreType>;

        int selectedItemIndex;
        ScoreType* maximaPtr;

        __device__
        UpdateMaxInLastColumnWithPointer(int which) : selectedItemIndex(which) {
            maximaPtr = new ScoreType[1];
            assert(maximaPtr != nullptr);
        }

        __device__
        ~UpdateMaxInLastColumnWithPointer(){
            delete [] maximaPtr;
        }

        __device__
        void operator()(ScoreType toCompare, int /*tileNr*/, int row, int itemIndex){
            if(itemIndex == selectedItemIndex){
                maximaPtr[0] = MathOps::max(maximaPtr[0], toCompare);
            }
        }
    };


    struct DoNotUpdateMaxInLastColumn{

        template<class ScoreType>
        __device__
        void operator()(ScoreType /*toCompare*/, int /*tileNr*/, int /*row*/, int /*itemIndex*/){
        }

        template<class ScoreType, int numItems>
        __device__
        void operator()(const ScoreType (&/*toCompare*/)[numItems], int /*tileNr*/, int /*row*/){
        }
    };

    template<class ScoreType>
    struct UpdateMaxInLastRow{
        using MathOps = MathOps<ScoreType>;

        ScoreType maximum{};

        __device__
        void operator()(ScoreType toCompare, int tileNr, int itemIndex){
            // printf("threadId %d, tile %d, itemIndex %d, compare with %f\n", threadIdx.x, tileNr, itemIndex, float(toCompare));
            // printf("threadId %d, tile %d, itemIndex %d, compare with (%f %f)\n", threadIdx.x, tileNr, itemIndex, float(toCompare.x), float(toCompare.y));
            maximum = MathOps::max(maximum, toCompare);
        }
    };

    // template<>
    // struct UpdateMaxInLastRow<half2>{
    //     using ScoreType = half2;
    //     using ScalarScoreType = ScalarScoreType<half2>;
    //     using MathOps = MathOps<ScalarScoreType>;

    //     ScalarScoreType maximum{};

    //     __device__
    //     void operator()(ScalarScoreType toCompare, int /*tileNr*/, int /*itemIndex*/){
    //         maximum = MathOps::max(maximum, toCompare);
    //     }
    // };

    template<class ScoreType>
    struct LastColumnLinear{
        using PayloadType = ScoreType;
        int targetThread;
        PayloadType M{};

        template<class Group>
        __device__
        void update(Group& group, int /*row*/, ScoreType Mnew){
            M = group.shfl_down(M, 1);
            if(group.thread_rank() == targetThread){
                M = Mnew;
            }
        }

        __device__
        PayloadType getPayload() const {
            return M;
        }
    };

    template<class ScoreType>
    struct LastColumnLinearLastThread{
        using PayloadType = ScoreType;
        PayloadType M{};

        template<class Group>
        __device__
        void update(Group& group, int /*row*/, ScoreType Mnew){
            M = group.shfl_down(M, 1);
            if(group.thread_rank() == (group.size()-1)){
                M = Mnew;
            }
        }

        __device__
        PayloadType getPayload() const {
            return M;
        }
    };

    template<class ScoreType>
    struct LastColumnAffine{
        using PayloadType = typename Vectorized2<ScoreType>::type;
        int targetThread;
        PayloadType vec_M_E;

        template<class Group>
        __device__
        void update(Group& group, int /*row*/, ScoreType M, ScoreType E){
            #if 1
            //work around issue with cooperative groups / half2
            static_assert(sizeof(vec_M_E) == sizeof(double));
            double tmp;
            memcpy(&tmp, &vec_M_E, sizeof(double));
            tmp = group.shfl_down(tmp, 1);
            memcpy(&vec_M_E, &tmp, sizeof(double));
            #else
            vec_M_E = group.shfl_down(vec_M_E, 1);
            #endif
            if(group.thread_rank() == targetThread){
                vec_M_E.x = M;
                vec_M_E.y = E;
            }
        }

        __device__
        PayloadType getPayload() const {
            return vec_M_E;
        }
    };

    template<class ScoreType>
    struct LastColumnAffineLastThread{
        using PayloadType = typename Vectorized2<ScoreType>::type;
        PayloadType vec_M_E;

        template<class Group>
        __device__
        void update(Group& group, int /*row*/, ScoreType M, ScoreType E){
            #if 1
            //work around issue with cooperative groups / half2
            static_assert(sizeof(vec_M_E) == sizeof(double));
            double tmp;
            memcpy(&tmp, &vec_M_E, sizeof(double));
            tmp = group.shfl_down(tmp, 1);
            memcpy(&vec_M_E, &tmp, sizeof(double));
            #else
            vec_M_E = group.shfl_down(vec_M_E, 1);
            #endif

            if(group.thread_rank() == (group.size()-1)){
                vec_M_E.x = M;
                vec_M_E.y = E;
            }
        }

        __device__
        PayloadType getPayload() const {
            return vec_M_E;
        }
    };

    struct NoLastColumn{
        template<class Group, class T1>
        __device__
        void update(Group& /*group*/, int /*row*/, T1 /*M*/){}

        template<class Group, class T1, class T2>
        __device__
        void update(Group& /*group*/, int /*row*/, T1 /*M*/, T2 /*E*/){}
    };


    template<class ScoreType>
    struct LeftBorderLinear{
        using PayloadType = ScoreType;
        PayloadType M;

        __device__
        ScoreType getM(int row, ScoreType gapscore) const{
            return M;
        }

        __device__
        void setPayload(PayloadType val){
            M = val;
        }

        __device__
        PayloadType getPayload() const {
            return M;
        }

        template<class Group>
        __device__
        void shuffleDown(Group& group){
            M = group.shfl_down(M, 1);
        }
    };

    template<class ScoreType>
    struct LeftBorderAffine{
        using PayloadType = typename Vectorized2<ScoreType>::type;
        PayloadType vec_M_E;

        __device__
        ScoreType getM(int row, ScoreType gapopenscore, ScoreType gapextendscore) const{
            return vec_M_E.x;
        }

        __device__
        ScoreType getE() const{
            return vec_M_E.y;
        }

        __device__
        void setPayload(PayloadType val){
            vec_M_E = val;
        }

        __device__
        PayloadType getPayload() const {
            return vec_M_E;
        }

        template<class Group>
        __device__
        void shuffleDown(Group& group){
            if constexpr(std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>){
                vec_M_E = group.shfl_down(vec_M_E, 1);
            }else{
                //work around issue with cooperative groups / half2
                static_assert(sizeof(PayloadType) == sizeof(double));
                double tmp;
                memcpy(&tmp, &vec_M_E, sizeof(double));
                tmp = group.shfl_down(tmp, 1);
                memcpy(&vec_M_E, &tmp, sizeof(double));
            }
        }
    };

    template<class ScoreType>
    struct FirstLeftBorder{
        __device__
        ScoreType getM(int row, ScoreType gapscore) const{
            return ScoreType{};
        }

        __device__
        ScoreType getM(int row, ScoreType gapopenscore, ScoreType gapextendscore) const{
            return ScoreType{};
        }

        __device__
        ScoreType getE() const{
            return OOBScore<ScoreType>::get();
        }
    };

}

#endif