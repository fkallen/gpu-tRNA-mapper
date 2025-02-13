#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda_fp16.h>
#include <thrust/device_malloc_allocator.h>

#ifndef SDIV
    #define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

enum class AlignmentType{
    GlobalAlignment,
    SemiglobalAlignment,
    LocalAlignment,
    Invalid
};

enum class PenaltyType{
    Linear,
    Affine
};

enum class RelaxMode{
    Four,
    Eight
};

enum class SubstitutionMatrixDimensionMode{
    SubjectSquaredQuerySquared,
    SubjectSquaredQueryLinear,
    SubjectLinearQuerySquared,
    SubjectLinearQueryLinear,
};

__inline__
std::string to_string(AlignmentType type){
    switch(type){
        case AlignmentType::GlobalAlignment: return "AlignmentType::GlobalAlignment";
        case AlignmentType::SemiglobalAlignment: return "AlignmentType::SemiglobalAlignment";
        case AlignmentType::LocalAlignment: return "AlignmentType::LocalAlignment";
        default: return "Invalid AlignmentType???";
    }
}

__inline__
std::string to_string(PenaltyType type){
    switch(type){
        case PenaltyType::Linear: return "PenaltyType::Linear";
        case PenaltyType::Affine: return "PenaltyType::Affine";
        default: return "Invalid PenaltyType???";
    }
}

template<class T, int numRows_, int numColumns_>
struct SharedSubstitutionMatrix{
    using value_type = T;
    
    static constexpr int numColumns = numColumns_;
    static constexpr int numRows = numRows_;

    __host__ __device__
    T* operator[](int row){
        return data[row];
    }
    __host__ __device__
    const T* operator[](int row) const{
        return data[row];
    }

    T data[numRows][numColumns];
};

template<class T> struct MultiplyWithInt;
template<> struct MultiplyWithInt<int>{ int __host__ __device__ operator()(int lhs, int rhs){ return lhs * rhs; }};
template<> struct MultiplyWithInt<float>{ float __host__ __device__ operator()(int lhs, float rhs){ return lhs * rhs; }};
template<> struct MultiplyWithInt<half2>{ half2 __host__ __device__ operator()(int lhs, half2 rhs){ return make_half2(lhs, lhs) * rhs; }};
template<> struct MultiplyWithInt<short2>{ short2 __host__ __device__ operator()(int lhs, short2 rhs){ return make_short2(lhs * rhs.x, lhs * rhs.y); }};



template<class T> struct OOBScore;
template<> struct OOBScore<int>{ __host__ __device__ static constexpr int get(){ return -999999; } };
template<> struct OOBScore<float>{ __host__ __device__ static constexpr float get(){ return -999999.f; } };
template<> struct OOBScore<half2>{ __host__ __device__ static half2 get(){ return make_half2(-9999.f, -9999.f); } };
template<> struct OOBScore<short2>{ __host__ __device__ static short2 get(){ return make_short2(-9999, -9999); } };



struct alignas(8) MyHalf22{
    half2 x;
    half2 y;
};

struct alignas(8) MyShort22{
    short2 x;
    short2 y;
};

struct alignas(16) MyHalf24{
    half2 x;
    half2 y;
    half2 z;
    half2 w;
};

struct alignas(16) MyShort24{
    short2 x;
    short2 y;
    short2 z;
    short2 w;
};

template<class T> struct Vectorized2;
template<> struct Vectorized2<half2>{ using type = MyHalf22; };
template<> struct Vectorized2<short2>{ using type = MyShort22; };
template<> struct Vectorized2<half>{ using type = half2; };
template<> struct Vectorized2<short>{ using type = short2; };
template<> struct Vectorized2<int>{ using type = int2; };
template<> struct Vectorized2<float>{ using type = float2; };

template<class T> struct Vectorized4;
template<> struct Vectorized4<half2>{ using type = MyHalf24; };
template<> struct Vectorized4<short2>{ using type = MyShort24; };
template<> struct Vectorized4<int>{ using type = int4; };
template<> struct Vectorized4<float>{ using type = float4; };

template<class ScoreType> struct ScalarScoreType{};
template<> struct ScalarScoreType<half2>{ using type = half; };
template<> struct ScalarScoreType<short2>{ using type = short; };
template<> struct ScalarScoreType<int>{ using type = int; };
template<> struct ScalarScoreType<float>{ using type = float; };

template<class ScoreType> struct ToScoreType16{};
template<> struct ToScoreType16<float>{ using type = half; };
template<> struct ToScoreType16<int>{ using type = short; };
template<> struct ToScoreType16<half>{ using type = half; };
template<> struct ToScoreType16<short>{ using type = short; };

template<class ScoreType> struct ToScoreType16x2{};
template<> struct ToScoreType16x2<float>{ using type = half2; };
template<> struct ToScoreType16x2<int>{ using type = short2; };
template<> struct ToScoreType16x2<half2>{ using type = half2; };
template<> struct ToScoreType16x2<short2>{ using type = short2; };

template<class ScoreType> struct ToScoreType32{};
template<> struct ToScoreType32<half2>{ using type = float; };
template<> struct ToScoreType32<short2>{ using type = int; };
template<> struct ToScoreType32<float>{ using type = float; };
template<> struct ToScoreType32<int>{ using type = int; };


template<class ScoreType, class U>
__host__ __device__
ScoreType make_vec2(U x, U y){
    using ScalarScoreType = typename ScalarScoreType<ScoreType>::type;
    ScalarScoreType xx = x;
    ScalarScoreType yy = y;
    return ScoreType{xx, yy};
}

struct int32dpx{};

template<class T> struct TypeString{ static constexpr const char* value = "unspecified"; };
template<> struct TypeString<int>{ static constexpr const char* value = "int"; };
template<> struct TypeString<short>{ static constexpr const char* value = "short"; };
template<> struct TypeString<char>{ static constexpr const char* value = "char"; };
template<> struct TypeString<float>{ static constexpr const char* value = "float"; };
template<> struct TypeString<int32dpx>{ static constexpr const char* value = "int32dpx"; };
template<> struct TypeString<short2>{ static constexpr const char* value = "short2"; };
template<> struct TypeString<half2>{ static constexpr const char* value = "half2"; };




struct EncodedAlignmentInputDataSingleQuery{
    static constexpr bool isSameQueryForAll = true;

    const std::int8_t* subjects;
    const size_t* subjectOffsets;
    const int* subjectLengths;
    const std::int8_t* query;
    int queryLength;
    int numAlignments;

    __host__ __device__
    int getNumAlignments() const{
        return numAlignments;
    }

    __host__ __device__
    const std::int8_t* getSubject(int i) const{
        return subjects + subjectOffsets[i];
    }

    __host__ __device__
    int getSubjectLength(int i) const{
        return subjectLengths[i];
    }

    __host__ __device__
    const std::int8_t* getQuery(int /*i*/) const{
        return query;
    }

    __host__ __device__
    int getQueryLength(int /*i*/) const{
        return queryLength;
    }

};

struct EncodedAlignmentInputDataSingleQueryConstantSubjectOffset{
    static constexpr bool isSameQueryForAll = true;

    const std::int8_t* subjects;
    size_t subjectPitchBytes;
    const int* subjectLengths;
    const std::int8_t* query;
    int queryLength;
    int numAlignments;

    __host__ __device__
    int getNumAlignments() const{
        return numAlignments;
    }

    __host__ __device__
    const std::int8_t* getSubject(int i) const{
        return subjects + subjectPitchBytes * i;
    }

    __host__ __device__
    int getSubjectLength(int i) const{
        return subjectLengths[i];
    }

    __host__ __device__
    const std::int8_t* getQuery(int /*i*/) const{
        return query;
    }

    __host__ __device__
    int getQueryLength(int /*i*/) const{
        return queryLength;
    }

};

struct EncodedAlignmentInputDataMultiQuery{
    static constexpr bool isSameQueryForAll = false;

    const std::int8_t* subjects;
    const size_t* subjectOffsets;
    const int* subjectLengths;
    const std::int8_t* queries;
    const size_t* queryOffsets;
    const int* queryLengths;
    int numAlignments;

    __host__ __device__
    int getNumAlignments() const{
        return numAlignments;
    }

    __host__ __device__
    const std::int8_t* getSubject(int i) const{
        return subjects + subjectOffsets[i];
    }

    __host__ __device__
    int getSubjectLength(int i) const{
        return subjectLengths[i];
    }

    __host__ __device__
    const std::int8_t* getQuery(int i) const{
        return queries + queryOffsets[i];
    }

    __host__ __device__
    int getQueryLength(int i) const{
        return queryLengths[i];
    }

};




struct Scoring1{
    int matchscore = 1;
    int mismatchscore = -1;
    int gapscore = -1;
    int gapopenscore = -4;
    int gapextendscore = -1;
};

template<class ScoreType>
struct ScoringKernelParam{
    ScoreType matchscore;
    ScoreType mismatchscore;
    ScoreType gapscore;
    ScoreType gapopenscore;
    ScoreType gapextendscore;

    __host__ __device__
    ScoringKernelParam(const Scoring1& s){
        if constexpr (std::is_same_v<ScoreType, float> || std::is_same_v<ScoreType, int>){
            matchscore = s.matchscore;
            mismatchscore = s.mismatchscore;
            gapscore = s.gapscore;
            gapopenscore = s.gapopenscore;
            gapextendscore = s.gapextendscore;
        }else if(std::is_same_v<ScoreType, half2> || std::is_same_v<ScoreType, short2>){
            matchscore = make_vec2<ScoreType>(s.matchscore,s.matchscore);
            mismatchscore = make_vec2<ScoreType>(s.mismatchscore,s.mismatchscore);
            gapscore = make_vec2<ScoreType>(s.gapscore,s.gapscore);
            gapopenscore = make_vec2<ScoreType>(s.gapopenscore,s.gapopenscore);
            gapextendscore = make_vec2<ScoreType>(s.gapextendscore,s.gapextendscore);
        }
    }

};






struct ConvertASCIIminus65{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        return c - 65;
    }
};





struct ConvertDNA{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'T') return 3;
        return 4;
    }
};

struct ConvertDNA4{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'T') return 3;
        return 4;
    }
};

struct ConvertDNA5{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'T') return 3;
        if(c == 'N') return 4;
        return 5;
    }
};

struct InverseConvertDNA4{
    __host__ __device__
    char operator()(const std::int8_t& AA) {
        if (AA == 0) return 'A';
        if (AA == 1) return 'C';
        if (AA == 2) return 'G';
        if (AA == 3) return 'T';
        return '#'; //  else
    }
};



struct ConvertAA_20{
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    constexpr std::int8_t operator()(const char& AA) {
        // ORDER of AminoAcids (NCBI): A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
        if (AA == 'A') return 0;
        if (AA == 'R') return 1;
        if (AA == 'N') return 2;
        if (AA == 'D') return 3;
        if (AA == 'C') return 4;
        if (AA == 'Q') return 5;
        if (AA == 'E') return 6;
        if (AA == 'G') return 7;
        if (AA == 'H') return 8;
        if (AA == 'I') return 9;
        if (AA == 'L') return 10;
        if (AA == 'K') return 11;
        if (AA == 'M') return 12;
        if (AA == 'F') return 13;
        if (AA == 'P') return 14;
        if (AA == 'S') return 15;
        if (AA == 'T') return 16;
        if (AA == 'W') return 17;
        if (AA == 'Y') return 18;
        if (AA == 'V') return 19;

        //lower-case has upper-case encoding
        if (AA == 'a') return 0;
        if (AA == 'r') return 1;
        if (AA == 'n') return 2;
        if (AA == 'd') return 3;
        if (AA == 'c') return 4;
        if (AA == 'q') return 5;
        if (AA == 'e') return 6;
        if (AA == 'g') return 7;
        if (AA == 'h') return 8;
        if (AA == 'i') return 9;
        if (AA == 'l') return 10;
        if (AA == 'k') return 11;
        if (AA == 'm') return 12;
        if (AA == 'f') return 13;
        if (AA == 'p') return 14;
        if (AA == 's') return 15;
        if (AA == 't') return 16;
        if (AA == 'w') return 17;
        if (AA == 'y') return 18;
        if (AA == 'v') return 19;

        return 20; //  else
    }
};



struct alignas(4) MyNuc4{
    __host__ __device__
    std::int8_t operator[](int i){
        if(i == 0) return x;
        else if(i == 1) return y;
        else if(i == 2) return z;
        else return w;
    }

    std::int8_t x;
    std::int8_t y;
    std::int8_t z;
    std::int8_t w;
};
static_assert(sizeof(MyNuc4) == sizeof(char4));




struct InverseConvertAA_20{
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    char operator()(const std::int8_t& AA) {
        // ORDER of AminoAcids (NCBI): A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
        if (AA == 0) return 'A';
        if (AA == 1) return 'R';
        if (AA == 2) return 'N';
        if (AA == 3) return 'D';
        if (AA == 4) return 'C';
        if (AA == 5) return 'Q';
        if (AA == 6) return 'E';
        if (AA == 7) return 'G';
        if (AA == 8) return 'H';
        if (AA == 9) return 'I';
        if (AA == 10) return 'L';
        if (AA == 11) return 'K';
        if (AA == 12) return 'M';
        if (AA == 13) return 'F';
        if (AA == 14) return 'P';
        if (AA == 15) return 'S';
        if (AA == 16) return 'T';
        if (AA == 17) return 'W';
        if (AA == 18) return 'Y';
        if (AA == 19) return 'V';
        return '-'; //  else
    }
};





template <class T>
struct thrust_async_allocator : public thrust::device_malloc_allocator<T> {
public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

    thrust_async_allocator(cudaStream_t stream_) : stream{stream_} {}

    pointer allocate(size_type num){
        //std::cout << "allocate " << num << "\n";
        T* result = nullptr;
        cudaError_t status = cudaMallocAsync(&result, sizeof(T) * num, stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error allocate");
        }
        return thrust::device_pointer_cast(result);
    }

    void deallocate(pointer ptr, size_type /*num*/){
        //std::cout << "deallocate \n";
        cudaError_t status = cudaFreeAsync(thrust::raw_pointer_cast(ptr), stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error deallocate");
        }
    }

private:
    cudaStream_t stream;
};















/*
    if begin <= i < end, call func(i) with i as compile time constant, else don't call func
*/
template<int begin, int end, class Func>
__host__ __device__ __forceinline__
void call_with_compile_time_constant_argument(Func func, int i){
    static_assert(begin <= end);

    if constexpr(begin == end){
        return;
    }else{
        if(begin == i){
            func(begin);
        }else{
            call_with_compile_time_constant_argument<begin+1,end>(func, i);
        }
    }
}

template<int begin, int end, class Func, class... Args>
__host__ __device__ __forceinline__
void call_with_compile_time_constant_argument_variadic(Func func, int i, Args&&... args){
    static_assert(begin <= end);

    if constexpr(begin == end){
        return;
    }else{
        if(begin == i){
            func(begin, std::forward<Args>(args)...);
        }else{
            call_with_compile_time_constant_argument_variadic<begin+1,end>(func, i, std::forward<Args>(args)...);
        }
    }
}












#endif
