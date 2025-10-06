#ifndef ALPHABET_CUH
#define ALPHABET_CUH

#include <cstdint>

constexpr int alphabetSize = 5; // A, C, G, T/U, N

struct ConvertLetters_alphabet_4{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'T' || c == 'U') return 3;
        return 0;
    }
};


struct ConvertLetters_alphabet_5{
    __host__ __device__
    constexpr std::int8_t operator()(char c){
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'T' || c == 'U') return 3;
        return 4;
    }
};


using ConvertLetters_functor = ConvertLetters_alphabet_5;



#endif