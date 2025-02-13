#ifndef MY_PARASAIL_HELPERS_HPP
#define MY_PARASAIL_HELPERS_HPP

#include <memory>

#include <parasail.h>


using ParasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>;
using ParasailResult = std::unique_ptr<parasail_result_t, void (*)(parasail_result_t*)>;
using ParasailMatrix = std::unique_ptr<parasail_matrix_t, void (*)(parasail_matrix_t*)>;

struct ParasailResultData{
    ParasailResultData() : result(nullptr, parasail_result_free), cigar(nullptr, parasail_cigar_free){}

    ParasailResult result;
    ParasailCigar cigar;
};

// struct ParasailResult{
//     ParasailResult() = default;
//     ParasailResult(const ParasailResult&) = delete;
//     ParasailResult(ParasailResult&& rhs){
//         data = std::exchange(rhs.data, nullptr);
//     }

//     ParasailResult(parasail_result_t* p) : data(p){}
//     ParasailResult(parasail_result_t* p, void (*)(parasail_result_t*)) : data(p){}

//     ParasailResult& operator=(const ParasailResult&) = delete;
//     ParasailResult& operator=(ParasailResult&& rhs){
//         if(data){
//             parasail_result_free(data);
//         }
//         data = std::exchange(rhs.data, nullptr);
//         return *this;
//     }

//     ~ParasailResult(){
//         if(data){
//             parasail_result_free(data);
//         }
//     }

//     parasail_result_t* get(){
//         return data;
//     }
//     parasail_result_t* get() const{
//         return data;
//     }

//     parasail_result_t* data{};
// };



using ParasailAlignmentFunction = parasail_result_t* (*)(
        const char *, int,
        const char *, int,
        int, int,
        const parasail_matrix_t*);


struct ParasailAlignmentTask{
    const char* s1{};
    int l1{};
    const char* s2{};
    int l2{};
    int gop{};
    int gex{};
    const parasail_matrix_t* matrix{};
    ParasailAlignmentFunction func = nullptr;

    ParasailAlignmentTask() = default;
    ParasailAlignmentTask(
        ParasailAlignmentFunction func_,
        const char * s1_, int s1Len_,
        const char * s2_, int s2Len_,
        int open_, int gap_,
        const parasail_matrix_t* matrix_
    ) :
        s1(s1_),
        l1(s1Len_),
        s2(s2_),
        l2(s2Len_),
        gop(open_),
        gex(gap_),
        matrix(matrix_),
        func(func_)
    {}

    ParasailResult compute() const{
        parasail_result_t* rawresult = func(s1,l1,s2,l2,gop,gex,matrix);
        return ParasailResult(rawresult, parasail_result_free);
    }
};






#endif