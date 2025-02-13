#ifndef LETTER_UTILITIES_CUH
#define LETTER_UTILITIES_CUH


#include "util.cuh"
#include <cstdint>




struct FuseTwoDNA4{

    __host__ __device__
    char4 fuseChar4_simple(char4 a, char4 b){
        return make_char4(
            a.x * 5 + b.x,
            a.y * 5 + b.y,
            a.z * 5 + b.z,
            a.w * 5 + b.w
        );
    }

    __host__ __device__
    char4 fuse(char4 a, char4 b) const{
        //compute per byte 5 * a + b
        unsigned int ua, ub;
        memcpy(&ua, &a, sizeof(unsigned int));
        memcpy(&ub, &b, sizeof(unsigned int));
        //dont need to AND 0xFCFCFCFC because the 2 most significant bits per byte are 0
        const unsigned int fourTimesA = (ua << 2); // & 0xFCFCFCFC
        const unsigned int uc = fourTimesA + ua + ub;
        char4 c;
        memcpy(&c, &uc, sizeof(unsigned int));
        return c;
    }

    __host__ __device__ 
    char4 operator()(char4 a, char4 b) const{
        return fuse(a,b);
    }

    __host__ __device__
    unsigned int fuse(unsigned int a, unsigned int b) const{
        //compute per byte 5 * a + b
        //dont need to AND 0xFCFCFCFC because the 2 most significant bits per byte are 0
        const unsigned int fourTimesA = (a << 2); // & 0xFCFCFCFC
        const unsigned int c = fourTimesA + a + b;
        return c;
    }

    __host__ __device__ 
    unsigned int operator()(unsigned int a, unsigned int b) const{
        return fuse(a,b);
    }

};



struct FuseTwoEncodedLetters{

    __host__ __device__
    int single(int a, int b, int DIM) const{
        return a * DIM + b;
    }

    __host__ __device__
    char4 char4_to_char4(char4 a, char4 b, int DIM) const{
        if(DIM == 4){
            unsigned int ua, ub;
            memcpy(&ua, &a, sizeof(unsigned int));
            memcpy(&ub, &b, sizeof(unsigned int));
            //dont need to AND 0xFCFCFCFC because the 2 most significant bits per byte are 0 (assuming each encoded letter is < 63)
            const unsigned int fourTimesA = (ua << 2); // & 0xFCFCFCFC
            const unsigned int uc = fourTimesA + ub;
            char4 c;
            memcpy(&c, &uc, sizeof(unsigned int));
            return c;
        }else if(DIM == 5){
            unsigned int ua, ub;
            memcpy(&ua, &a, sizeof(unsigned int));
            memcpy(&ub, &b, sizeof(unsigned int));
            //dont need to AND 0xFCFCFCFC because the 2 most significant bits per byte are 0 (assuming each encoded letter is < 63)
            const unsigned int fourTimesA = (ua << 2); // & 0xFCFCFCFC
            const unsigned int uc = fourTimesA + ua + ub;
            char4 c;
            memcpy(&c, &uc, sizeof(unsigned int));
            return c;
        }else{
            return make_char4(
                single(a.x, b.x, DIM),
                single(a.y, b.y, DIM),
                single(a.z, b.z, DIM),
                single(a.w, b.w, DIM)
            );
        }
        
    }

    __host__ __device__
    uchar4 char4_to_uchar4(char4 a, char4 b, int DIM) const{
        return make_uchar4(
            single(a.x, b.x, DIM),
            single(a.y, b.y, DIM),
            single(a.z, b.z, DIM),
            single(a.w, b.w, DIM)
        );
    }

    __host__ __device__
    int4 char4_to_int4(char4 a, char4 b, int DIM) const{
        return make_int4(
            single(a.x, b.x, DIM),
            single(a.y, b.y, DIM),
            single(a.z, b.z, DIM),
            single(a.w, b.w, DIM)
        );
    }

};


template<class Group, int oobLetter>
struct SubjectLettersData_v1{
public:
    __device__
    SubjectLettersData_v1(Group& g, const std::int8_t* ptr, int length)
        : subjectLoadOffset(g.thread_rank()), 
        subjectLength(length), 
        loadOffsetLimit(SDIV(subjectLength, 4)),
        subjectData(ptr), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        if(subjectLoadOffset < loadOffsetLimit){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData);
            current4Letters = subjectAsChar4[subjectLoadOffset];
            subjectLoadOffset += group.size();
        }else{
            current4Letters = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }       
        // if(subjectLoadOffset < loadOffsetLimit){
        //     if(size_t(subjectData) % 4 == 0){
        //         const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData);
        //         current4Letters = subjectAsChar4[subjectLoadOffset];
        //     }else{
        //         current4Letters.x = subjectData[4*subjectLoadOffset+0];
        //         current4Letters.y = subjectData[4*subjectLoadOffset+1];
        //         current4Letters.z = subjectData[4*subjectLoadOffset+2];
        //         current4Letters.w = subjectData[4*subjectLoadOffset+3];
        //     }
        //     subjectLoadOffset += group.size();
        // }else{
        //     current4Letters = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        //     // current4Letters = make_int4(oobLetter, oobLetter, oobLetter, oobLetter);
        // }
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));

        // group.shfl_down(current4Letters, 1);
    }

private:
    char4 current4Letters;
    // int4 current4Letters;
    int currentLetter = oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const int loadOffsetLimit;
    const std::int8_t* const subjectData;
    Group& group;            
};


template<class Group, int oobLetter>
struct SubjectLettersData_v2{
public:
    __device__
    SubjectLettersData_v2(Group& g, const std::int8_t* ptr, int length)
        : subjectLoadOffset(4*g.thread_rank()), 
        subjectLength(length), 
        subjectData(ptr), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        if(subjectLoadOffset / 4 < subjectLength / 4){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData);
            current4Letters = subjectAsChar4[subjectLoadOffset / 4];
            subjectLoadOffset += 4*group.size();
        }else if(subjectLoadOffset < subjectLength){
            current4Letters.x = subjectData[subjectLoadOffset];
            current4Letters.y = (subjectLoadOffset+1 < subjectLength) ? subjectData[subjectLoadOffset+1] : oobLetter;
            current4Letters.z = (subjectLoadOffset+2 < subjectLength) ? subjectData[subjectLoadOffset+2] : oobLetter;
            current4Letters.w = (subjectLoadOffset+3 < subjectLength) ? subjectData[subjectLoadOffset+3] : oobLetter;
            subjectLoadOffset += 4*group.size();
        }else{
            current4Letters = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }       
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));

        // group.shfl_down(current4Letters, 1);
    }

private:
    char4 current4Letters;
    // int4 current4Letters;
    int currentLetter = oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const std::int8_t* const subjectData;
    Group& group;
};



template<class Group, int oobLetter>
struct SubjectLettersData_v3{
public:
    __device__
    SubjectLettersData_v3(Group& g, const std::int8_t* ptr, int length)
        : subjectLoadOffset(4*g.thread_rank()), 
        subjectLength(length), 
        subjectData(ptr), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        if(subjectLoadOffset / 4 < subjectLength / 4){
            if(size_t(subjectData) % sizeof(char4) == 0){
                const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData);
                current4Letters = subjectAsChar4[subjectLoadOffset / 4];
            }else{
                current4Letters.x = subjectData[subjectLoadOffset];
                current4Letters.y = subjectData[subjectLoadOffset+1];
                current4Letters.z = subjectData[subjectLoadOffset+2];
                current4Letters.w = subjectData[subjectLoadOffset+3];
            }
            subjectLoadOffset += 4*group.size();
        }else if(subjectLoadOffset < subjectLength){
            current4Letters.x = subjectData[subjectLoadOffset];
            current4Letters.y = (subjectLoadOffset+1 < subjectLength) ? subjectData[subjectLoadOffset+1] : oobLetter;
            current4Letters.z = (subjectLoadOffset+2 < subjectLength) ? subjectData[subjectLoadOffset+2] : oobLetter;
            current4Letters.w = (subjectLoadOffset+3 < subjectLength) ? subjectData[subjectLoadOffset+3] : oobLetter;
            subjectLoadOffset += 4*group.size();
        }else{
            current4Letters = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }       
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));

        // group.shfl_down(current4Letters, 1);
    }

private:
    char4 current4Letters;
    // int4 current4Letters;
    int currentLetter = oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const std::int8_t* const subjectData;
    Group& group;            
};


template<class Group, int oobLetter>
struct SubjectLettersData_v4{
public:
    __device__
    SubjectLettersData_v4(Group& g, const std::int8_t* ptr, int length)
        : subjectLoadOffset(4*g.thread_rank()), 
        subjectLength(length), 
        subjectData(ptr), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        current4Letters = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        if(subjectLoadOffset < subjectLength){
            current4Letters.x = subjectData[subjectLoadOffset];
        }
        if(subjectLoadOffset+1 < subjectLength){
            current4Letters.y = subjectData[subjectLoadOffset+1];
        }
        if(subjectLoadOffset+2 < subjectLength){
            current4Letters.z = subjectData[subjectLoadOffset+2];
        }
        if(subjectLoadOffset+3 < subjectLength){
            current4Letters.w = subjectData[subjectLoadOffset+3];
        }
        subjectLoadOffset += 4*group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));

        // group.shfl_down(current4Letters, 1);
    }

private:
    char4 current4Letters;
    // int4 current4Letters;
    int currentLetter = oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const std::int8_t* const subjectData;
    Group& group;
};








template<int alphabetSizeIncludingExtraLetters, class Group>
struct SubjectPairLettersData_smallAlphabet_v1{
public:
    static constexpr int oobLetter = alphabetSizeIncludingExtraLetters-1;
    static constexpr int fuseDim = alphabetSizeIncludingExtraLetters;
    static_assert(oobLetter * fuseDim + oobLetter <= 127);

    __device__
    SubjectPairLettersData_smallAlphabet_v1(
        Group& g, 
        const std::int8_t* ptr0, 
        int length0,
        const std::int8_t* ptr1, 
        int length1
    )
        : subjectLoadOffset(g.thread_rank()), 
        // oobLetter(oobLetter_),
        // fuseDim(fuseDim_),
        currentLetter(FuseTwoEncodedLetters{}.single(oobLetter,oobLetter,fuseDim)),
        loadOffsetLimit0(SDIV(length0, 4)),
        loadOffsetLimit1(SDIV(length1, 4)),
        subjectData0(ptr0), 
        subjectData1(ptr1), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        char4 current4Letters0;
        char4 current4Letters1;
        if(subjectLoadOffset < loadOffsetLimit0){
            const char4* const subject0AsChar4 = reinterpret_cast<const char4*>(subjectData0);
            current4Letters0 = subject0AsChar4[subjectLoadOffset];
        }else{
            current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }
        if(subjectLoadOffset < loadOffsetLimit1){
            const char4* const subject1AsChar4 = reinterpret_cast<const char4*>(subjectData1);
            current4Letters1 = subject1AsChar4[subjectLoadOffset];
        }else{
            current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        current4Letters = FuseTwoEncodedLetters{}.char4_to_char4(current4Letters0, current4Letters1, fuseDim);

        subjectLoadOffset += group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));
    }

private:
    int subjectLoadOffset;
    // const int oobLetter;
    // const int fuseDim;
    int currentLetter;
    char4 current4Letters;
    const int loadOffsetLimit0;
    const int loadOffsetLimit1;
    const std::int8_t* const subjectData0;
    const std::int8_t* const subjectData1;
    Group& group;            
};




template<int alphabetSizeIncludingExtraLetters, class Group>
struct SubjectPairLettersData_smallAlphabet_v2{
public:
    static constexpr int oobLetter = alphabetSizeIncludingExtraLetters-1;
    static constexpr int fuseDim = alphabetSizeIncludingExtraLetters;
    static_assert(oobLetter * fuseDim + oobLetter <= 127);

    __device__
    SubjectPairLettersData_smallAlphabet_v2(
        Group& g, 
        const std::int8_t* ptr0, 
        int length0,
        const std::int8_t* ptr1, 
        int length1
    )
        : subjectLoadOffset(4*g.thread_rank()), 
        // oobLetter(oobLetter_),
        // fuseDim(fuseDim_),
        currentLetter(FuseTwoEncodedLetters{}.single(oobLetter,oobLetter,fuseDim)),
        subjectLength0(length0),
        subjectLength1(length1),
        subjectData0(ptr0), 
        subjectData1(ptr1), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        char4 current4Letters0;
        char4 current4Letters1;

        #if 0
        //for some reason this code loses 10% performance for substitution matrix kernels with half2 (rtx4090, cuda12.6)
        //pssm is unaffected

        if(subjectLoadOffset / 4 < subjectLength0 / 4){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData0);
            current4Letters0 = subjectAsChar4[subjectLoadOffset / 4];
        }else if(subjectLoadOffset < subjectLength0){
            current4Letters0.x = subjectData0[subjectLoadOffset];
            current4Letters0.y = (subjectLoadOffset+1 < subjectLength0) ? subjectData0[subjectLoadOffset+1] : oobLetter;
            current4Letters0.z = (subjectLoadOffset+2 < subjectLength0) ? subjectData0[subjectLoadOffset+2] : oobLetter;
            current4Letters0.w = (subjectLoadOffset+3 < subjectLength0) ? subjectData0[subjectLoadOffset+3] : oobLetter;
        }else{
            current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        if(subjectLoadOffset / 4 < subjectLength1 / 4){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData1);
            current4Letters1 = subjectAsChar4[subjectLoadOffset / 4];
        }else if(subjectLoadOffset < subjectLength1){
            current4Letters1.x = subjectData1[subjectLoadOffset];
            current4Letters1.y = (subjectLoadOffset+1 < subjectLength1) ? subjectData1[subjectLoadOffset+1] : oobLetter;
            current4Letters1.z = (subjectLoadOffset+2 < subjectLength1) ? subjectData1[subjectLoadOffset+2] : oobLetter;
            current4Letters1.w = (subjectLoadOffset+3 < subjectLength1) ? subjectData1[subjectLoadOffset+3] : oobLetter;
        }else{
            current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        #else 
        //this code does not have above performance problemes

        current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        if(subjectLoadOffset < subjectLength0){
            current4Letters0.x = subjectData0[subjectLoadOffset];
        }
        if(subjectLoadOffset+1 < subjectLength0){
            current4Letters0.y = subjectData0[subjectLoadOffset+1];
        }
        if(subjectLoadOffset+2 < subjectLength0){
            current4Letters0.z = subjectData0[subjectLoadOffset+2];
        }
        if(subjectLoadOffset+3 < subjectLength0){
            current4Letters0.w = subjectData0[subjectLoadOffset+3];
        }

        current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        if(subjectLoadOffset < subjectLength1){
            current4Letters1.x = subjectData1[subjectLoadOffset];
        }
        if(subjectLoadOffset+1 < subjectLength1){
            current4Letters1.y = subjectData1[subjectLoadOffset+1];
        }
        if(subjectLoadOffset+2 < subjectLength1){
            current4Letters1.z = subjectData1[subjectLoadOffset+2];
        }
        if(subjectLoadOffset+3 < subjectLength1){
            current4Letters1.w = subjectData1[subjectLoadOffset+3];
        }
       

        #endif



        current4Letters = FuseTwoEncodedLetters{}.char4_to_char4(current4Letters0, current4Letters1, fuseDim);

        // printf("%d %d %d %d, %d %d %d %d -> %d %d %d %d\n",
        //     int(current4Letters0.x),
        //     int(current4Letters0.y),
        //     int(current4Letters0.z),
        //     int(current4Letters0.w),
        //     int(current4Letters1.x),
        //     int(current4Letters1.y),
        //     int(current4Letters1.z),
        //     int(current4Letters1.w),
        //     int(current4Letters.x),
        //     int(current4Letters.y),
        //     int(current4Letters.z),
        //     int(current4Letters.w)
        // );
        subjectLoadOffset += 4*group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));
    }

private:
    int subjectLoadOffset;
    // const int oobLetter;
    // const int fuseDim;
    int currentLetter;
    char4 current4Letters;
    const int subjectLength0;
    const int subjectLength1;
    const std::int8_t* const subjectData0;
    const std::int8_t* const subjectData1;
    Group& group;            
};



template<int alphabetSizeIncludingExtraLetters, class Group>
struct SubjectPairLettersData_largeAlphabet_v1{
public:
    static constexpr int oobLetter = alphabetSizeIncludingExtraLetters-1;
    static constexpr int fuseDim = alphabetSizeIncludingExtraLetters;

    __device__
    SubjectPairLettersData_largeAlphabet_v1(
        Group& g, 
        const std::int8_t* ptr0, 
        int length0,
        const std::int8_t* ptr1, 
        int length1
    )
        : subjectLoadOffset(g.thread_rank()), 
        // oobLetter(oobLetter_),
        // fuseDim(fuseDim_),
        currentLetter(FuseTwoEncodedLetters{}.single(oobLetter,oobLetter,fuseDim)),
        loadOffsetLimit0(SDIV(length0, 4)),
        loadOffsetLimit1(SDIV(length1, 4)),
        subjectData0(ptr0), 
        subjectData1(ptr1), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        char4 current4Letters0;
        char4 current4Letters1;
        if(subjectLoadOffset < loadOffsetLimit0){
            const char4* const subject0AsChar4 = reinterpret_cast<const char4*>(subjectData0);
            current4Letters0 = subject0AsChar4[subjectLoadOffset];
        }else{
            current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }
        if(subjectLoadOffset < loadOffsetLimit1){
            const char4* const subject1AsChar4 = reinterpret_cast<const char4*>(subjectData1);
            current4Letters1 = subject1AsChar4[subjectLoadOffset];
        }else{
            current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        current4Letters = FuseTwoEncodedLetters{}.char4_to_int4(current4Letters0, current4Letters1, fuseDim);

        subjectLoadOffset += group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        current4Letters = group.shfl_down(current4Letters, 1);
    }

private:
    int subjectLoadOffset;
    // const int oobLetter;
    // const int fuseDim;
    int currentLetter;
    const int loadOffsetLimit0;
    const int loadOffsetLimit1;
    const std::int8_t* const subjectData0;
    const std::int8_t* const subjectData1;
    int4 current4Letters;
    Group& group;            
};




template<int alphabetSizeIncludingExtraLetters, class Group>
struct SubjectPairLettersData_largeAlphabet_v2{
public:
    static constexpr int oobLetter = alphabetSizeIncludingExtraLetters-1;
    static constexpr int fuseDim = alphabetSizeIncludingExtraLetters;

    __device__
    SubjectPairLettersData_largeAlphabet_v2(
        Group& g, 
        const std::int8_t* ptr0, 
        int length0,
        const std::int8_t* ptr1, 
        int length1
    )
        : subjectLoadOffset(4*g.thread_rank()), 
        // oobLetter(oobLetter_),
        // fuseDim(fuseDim_),
        currentLetter(FuseTwoEncodedLetters{}.single(oobLetter,oobLetter,fuseDim)),
        subjectLength0(length0),
        subjectLength1(length1),
        subjectData0(ptr0), 
        subjectData1(ptr1), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        char4 current4Letters0;
        char4 current4Letters1;

        #if 0
        //for some reason this code loses 15% performance for substitution matrix kernels with half2 (rtx4090, cuda12.6)
        //pssm is unaffected

        if(subjectLoadOffset / 4 < subjectLength0 / 4){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData0);
            current4Letters0 = subjectAsChar4[subjectLoadOffset / 4];
        }else if(subjectLoadOffset < subjectLength0){
            current4Letters0.x = subjectData0[subjectLoadOffset];
            current4Letters0.y = (subjectLoadOffset+1 < subjectLength0) ? subjectData0[subjectLoadOffset+1] : oobLetter;
            current4Letters0.z = (subjectLoadOffset+2 < subjectLength0) ? subjectData0[subjectLoadOffset+2] : oobLetter;
            current4Letters0.w = (subjectLoadOffset+3 < subjectLength0) ? subjectData0[subjectLoadOffset+3] : oobLetter;
        }else{
            current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        if(subjectLoadOffset / 4 < subjectLength1 / 4){
            const char4* const subjectAsChar4 = reinterpret_cast<const char4*>(subjectData1);
            current4Letters1 = subjectAsChar4[subjectLoadOffset / 4];
        }else if(subjectLoadOffset < subjectLength1){
            current4Letters1.x = subjectData1[subjectLoadOffset];
            current4Letters1.y = (subjectLoadOffset+1 < subjectLength1) ? subjectData1[subjectLoadOffset+1] : oobLetter;
            current4Letters1.z = (subjectLoadOffset+2 < subjectLength1) ? subjectData1[subjectLoadOffset+2] : oobLetter;
            current4Letters1.w = (subjectLoadOffset+3 < subjectLength1) ? subjectData1[subjectLoadOffset+3] : oobLetter;
        }else{
            current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        #else 
        //this code does not have above performance problemes

        current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        if(subjectLoadOffset < subjectLength0){
            current4Letters0.x = subjectData0[subjectLoadOffset];
        }
        if(subjectLoadOffset+1 < subjectLength0){
            current4Letters0.y = subjectData0[subjectLoadOffset+1];
        }
        if(subjectLoadOffset+2 < subjectLength0){
            current4Letters0.z = subjectData0[subjectLoadOffset+2];
        }
        if(subjectLoadOffset+3 < subjectLength0){
            current4Letters0.w = subjectData0[subjectLoadOffset+3];
        }

        current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        if(subjectLoadOffset < subjectLength1){
            current4Letters1.x = subjectData1[subjectLoadOffset];
        }
        if(subjectLoadOffset+1 < subjectLength1){
            current4Letters1.y = subjectData1[subjectLoadOffset+1];
        }
        if(subjectLoadOffset+2 < subjectLength1){
            current4Letters1.z = subjectData1[subjectLoadOffset+2];
        }
        if(subjectLoadOffset+3 < subjectLength1){
            current4Letters1.w = subjectData1[subjectLoadOffset+3];
        }
       

        #endif



        current4Letters = FuseTwoEncodedLetters{}.char4_to_int4(current4Letters0, current4Letters1, fuseDim);

        // printf("%d %d %d %d, %d %d %d %d -> %d %d %d %d\n",
        //     int(current4Letters0.x),
        //     int(current4Letters0.y),
        //     int(current4Letters0.z),
        //     int(current4Letters0.w),
        //     int(current4Letters1.x),
        //     int(current4Letters1.y),
        //     int(current4Letters1.z),
        //     int(current4Letters1.w),
        //     int(current4Letters.x),
        //     int(current4Letters.y),
        //     int(current4Letters.z),
        //     int(current4Letters.w)
        // );
        subjectLoadOffset += 4*group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        current4Letters = group.shfl_down(current4Letters, 1);
    }

private:
    int subjectLoadOffset;
    // const int oobLetter;
    // const int fuseDim;
    int currentLetter;
    const int subjectLength0;
    const int subjectLength1;
    const std::int8_t* const subjectData0;
    const std::int8_t* const subjectData1;
    int4 current4Letters;
    Group& group;            
};



template<class Group, int oobLetter>
using SubjectLettersData = SubjectLettersData_v4<Group, oobLetter>;

// template<class Group, int oobLetter>
// using SubjectPairLettersData = SubjectPairLettersData_v1<Group, oobLetter>;

template<int alphabetSizeIncludingExtraLetters, class Group>
using SubjectPairLettersData = typename std::conditional<
    alphabetSizeIncludingExtraLetters <= 10,
    SubjectPairLettersData_smallAlphabet_v2<alphabetSizeIncludingExtraLetters, Group>,
    SubjectPairLettersData_largeAlphabet_v2<alphabetSizeIncludingExtraLetters, Group>
>::type;

















template<class Group, int oobLetter>
struct SameLengthSubjectPairLettersData{
public:
    __device__
    SameLengthSubjectPairLettersData(
        Group& g, 
        const std::int8_t* ptr0, 
        int length0,
        const std::int8_t* ptr1, 
        int length1
    )
        : subjectLoadOffset(g.thread_rank()), 
        subjectLength(length0), 
        loadOffsetLimit(SDIV(subjectLength, 4)),
        subjectData0(ptr0), 
        subjectData1(ptr1), 
        group(g)
    {
        assert(length0 == length1);
    }

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 4);
        if constexpr(Index == 0){
            currentLetter = current4Letters.x;
        }else if(Index == 1){
            currentLetter = current4Letters.y;
        }else if(Index == 2){
            currentLetter = current4Letters.z;
        }else{
            currentLetter = current4Letters.w;
        }
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext4Letters(){
        char4 current4Letters0;
        char4 current4Letters1;
        if(subjectLoadOffset < loadOffsetLimit){
            const char4* const subject0AsChar4 = reinterpret_cast<const char4*>(subjectData0);
            current4Letters0 = subject0AsChar4[subjectLoadOffset];
        }else{
            current4Letters0 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }
        if(subjectLoadOffset < loadOffsetLimit){
            const char4* const subject1AsChar4 = reinterpret_cast<const char4*>(subjectData1);
            current4Letters1 = subject1AsChar4[subjectLoadOffset];
        }else{
            current4Letters1 = make_char4(oobLetter, oobLetter, oobLetter, oobLetter);
        }

        FuseTwoDNA4 fuseChar4;
        current4Letters = fuseChar4(current4Letters0, current4Letters1);
        subjectLoadOffset += group.size();
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffle4Letters(){
        static_assert(sizeof(char4) == sizeof(int));
        int temp;
        memcpy(&temp, &current4Letters, sizeof(char4));
        temp = group.shfl_down(temp, 1);
        memcpy(&current4Letters, &temp, sizeof(int));
    }

private:
    char4 current4Letters;
    int currentLetter = oobLetter * (oobLetter+1) + oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const int loadOffsetLimit;
    const std::int8_t* const subjectData0;
    const std::int8_t* const subjectData1;
    Group& group;            
};





// template<class Group, int oobLetter>
// struct SubjectSingleLettersData{
// public:
//     __device__
//     SubjectSingleLettersData(Group& g, const std::int8_t* ptr, int length)
//         : subjectLoadOffset(g.thread_rank()), 
//         subjectLength(length), 
//         subjectData(ptr), 
//         group(g)
//     {}

//     __device__
//     int getCurrentLetter() const{
//         return currentLetter;
//     }

//     __device__
//     void loadNextLetter(){
//         if(subjectLoadOffset < subjectLength){
//             currentLetter = subjectData[subjectLoadOffset];
//             subjectLoadOffset += group.size();
//         }else{
//             currentLetter = oobLetter;
//         }
//     }

//     __device__
//     void shuffleCurrentLetter(){
//         currentLetter = group.shfl_up(currentLetter, 1);
//     }


// private:
//     int currentLetter = oobLetter;
//     int subjectLoadOffset;
//     const int subjectLength;
//     const std::int8_t* const subjectData;
//     Group& group;            
// };

template<class Group, int oobLetter>
struct SubjectSingleLettersData{
public:
    __device__
    SubjectSingleLettersData(Group& g, const std::int8_t* ptr, int length)
        : subjectLoadOffset(g.thread_rank()), 
        subjectLength(length), 
        subjectData(ptr), 
        group(g)
    {}

    template<int Index>
    __device__
    void setCurrentLetter(){
        static_assert(0 <= Index && Index < 1);
        currentLetter = currentCachedLetter;
    }

    __device__
    int getCurrentLetter() const{
        return currentLetter;
    }

    __device__
    void loadNext1Letter(){
        if(subjectLoadOffset < subjectLength){
            currentCachedLetter = subjectData[subjectLoadOffset];
            subjectLoadOffset += group.size();
        }else{
            currentCachedLetter = oobLetter;
        }
    }

    __device__
    void shuffleCurrentLetter(){
        currentLetter = group.shfl_up(currentLetter, 1);
    }

    __device__
    void shuffleCachedLetters(){
        currentCachedLetter = group.shfl_down(currentCachedLetter, 1);
    }

private:
    int currentCachedLetter;
    int currentLetter = oobLetter;
    int subjectLoadOffset;
    const int subjectLength;
    const std::int8_t* const subjectData;
    Group& group;            
};



#endif