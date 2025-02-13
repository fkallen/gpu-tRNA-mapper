#ifndef SUBSTITUTION_MATRIX_CUH
#define SUBSTITUTION_MATRIX_CUH

#include "util.cuh"
#include "cuda_errorcheck.cuh"

#include <type_traits>

#include <thrust/device_vector.h>


template<AlignmentType alignmentType_, class ScoreType32_, int Dim_>
struct GpuSubstitutionMatrix{
    static_assert(std::is_same_v<ScoreType32_, int> || std::is_same_v<ScoreType32_, float>);

    using ScoreType32 = ScoreType32_;
    using ScoreType16x2 = typename ToScoreType16x2<ScoreType32>::type;
    using ScoreType16 = typename ToScoreType16<ScoreType32>::type;

    static constexpr AlignmentType alignmentType = alignmentType_;
    static constexpr int Dim = Dim_;

    using Submat32 = SharedSubstitutionMatrix<ScoreType32, Dim, Dim>;
    using Submat16x2_SubjectSquaredQuerySquared = SharedSubstitutionMatrix<ScoreType16x2, Dim*Dim, Dim*Dim>;
    using Submat16x2_SubjectSquaredQueryLinear = SharedSubstitutionMatrix<ScoreType16x2, Dim*Dim, Dim>;
    using Submat16x2_SubjectLinearQuerySquared = SharedSubstitutionMatrix<ScoreType16x2, Dim, Dim*Dim>;
    using Submat16x2_SubjectLinearQueryLinear = SharedSubstitutionMatrix<ScoreType16, Dim, Dim>;

    GpuSubstitutionMatrix() 
        : submat32(sizeof(Submat32)), 
        submat16x2_SubjectSquaredQuerySquared(sizeof(Submat16x2_SubjectSquaredQuerySquared)), 
        submat16x2_SubjectSquaredQueryLinear(sizeof(Submat16x2_SubjectSquaredQueryLinear)), 
        submat16x2_SubjectLinearQuerySquared(sizeof(Submat16x2_SubjectLinearQuerySquared)), 
        submat16x2_SubjectLinearQueryLinear(sizeof(Submat16x2_SubjectLinearQueryLinear))
    {
        size_t bytes = submat32.size() 
            + submat16x2_SubjectSquaredQuerySquared.size() 
            + submat16x2_SubjectSquaredQueryLinear.size() 
            + submat16x2_SubjectLinearQuerySquared.size() 
            + submat16x2_SubjectLinearQueryLinear.size();
        // std::cout << "GpuSubstitutionMatrix " << bytes << "\n";
    }

    GpuSubstitutionMatrix(const GpuSubstitutionMatrix&) = default;
    GpuSubstitutionMatrix(GpuSubstitutionMatrix&&) = default;
    GpuSubstitutionMatrix& operator=(const GpuSubstitutionMatrix&) = default;
    GpuSubstitutionMatrix& operator=(GpuSubstitutionMatrix&&) = default;

    Submat32* getSubmat32(){
        return reinterpret_cast<Submat32*>(submat32.data().get());
    }

    Submat16x2_SubjectSquaredQuerySquared* getSubmat16x2_SubjectSquaredQuerySquared(){
        return reinterpret_cast<Submat16x2_SubjectSquaredQuerySquared*>(submat16x2_SubjectSquaredQuerySquared.data().get());
    }

    Submat16x2_SubjectSquaredQueryLinear* getSubmat16x2_SubjectSquaredQueryLinear(){
        return reinterpret_cast<Submat16x2_SubjectSquaredQueryLinear*>(submat16x2_SubjectSquaredQueryLinear.data().get());
    }

    Submat16x2_SubjectLinearQuerySquared* getSubmat16x2_SubjectLinearQuerySquared(){
        return reinterpret_cast<Submat16x2_SubjectLinearQuerySquared*>(submat16x2_SubjectLinearQuerySquared.data().get());
    }

    Submat16x2_SubjectLinearQueryLinear* getSubmat16x2_SubjectLinearQueryLinear(){
        return reinterpret_cast<Submat16x2_SubjectLinearQueryLinear*>(submat16x2_SubjectLinearQueryLinear.data().get());
    }

    const Submat32* getSubmat32() const {
        return reinterpret_cast<const Submat32*>(submat32.data().get());
    }

    const Submat16x2_SubjectSquaredQuerySquared* getSubmat16x2_SubjectSquaredQuerySquared() const {
        return reinterpret_cast<const Submat16x2_SubjectSquaredQuerySquared*>(submat16x2_SubjectSquaredQuerySquared.data().get());
    }

    const Submat16x2_SubjectSquaredQueryLinear* getSubmat16x2_SubjectSquaredQueryLinear() const {
        return reinterpret_cast<const Submat16x2_SubjectSquaredQueryLinear*>(submat16x2_SubjectSquaredQueryLinear.data().get());
    }

    const Submat16x2_SubjectLinearQuerySquared* getSubmat16x2_SubjectLinearQuerySquared() const {
        return reinterpret_cast<const Submat16x2_SubjectLinearQuerySquared*>(submat16x2_SubjectLinearQuerySquared.data().get());
    }

    const Submat16x2_SubjectLinearQueryLinear* getSubmat16x2_SubjectLinearQueryLinear() const {
        return reinterpret_cast<const Submat16x2_SubjectLinearQueryLinear*>(submat16x2_SubjectLinearQueryLinear.data().get());
    }

    thrust::device_vector<char> submat32;
    thrust::device_vector<char> submat16x2_SubjectSquaredQuerySquared;
    thrust::device_vector<char> submat16x2_SubjectSquaredQueryLinear;
    thrust::device_vector<char> submat16x2_SubjectLinearQuerySquared;
    thrust::device_vector<char> submat16x2_SubjectLinearQueryLinear;
};

template<class ScoreType, int Dim>
auto makeGpuSubstitutionMatrix_localalignment(
    const int* h_substitutionMatrix, //Dim * Dim,  [targetLetter * Dim + queryLetter]
    cudaStream_t stream
){
    using ScoreType32 = typename ToScoreType32<ScoreType>::type;
    constexpr AlignmentType alignmentType = AlignmentType::LocalAlignment;
    using GpuMatrix = GpuSubstitutionMatrix<alignmentType, ScoreType32, Dim+1>;
    GpuMatrix gpuSubstitutionMatrix;

    typename GpuMatrix::Submat32 substitutionMatrix32;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            substitutionMatrix32.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }
    //padding for local alignment
    for(int x = 0; x < Dim+1; x++){
        substitutionMatrix32.data[Dim][x] = 0;
    }
    for(int y = 0; y < Dim+1; y++){
        substitutionMatrix32.data[y][Dim] = 0;
    }
    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat32(), 
        &substitutionMatrix32, 
        sizeof(typename GpuMatrix::Submat32),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared submat16x2_SubjectSquaredQuerySquared;

    for(int y = 0; y < (Dim+1)*(Dim+1); y++){
        const int leftY = y / (Dim+1);
        const int rightY = y % (Dim+1);
        for(int x = 0; x < (Dim+1)*(Dim+1); x++){
            const int leftX = x / (Dim+1);
            const int rightX = x % (Dim+1);
            if(leftY < Dim && leftX < Dim){
                submat16x2_SubjectSquaredQuerySquared.data[y][x].x = h_substitutionMatrix[leftY * Dim + leftX];
            }else{
                submat16x2_SubjectSquaredQuerySquared.data[y][x].x = 0; //padding for local alignment
            }
            if(rightY < Dim && rightX < Dim){
                submat16x2_SubjectSquaredQuerySquared.data[y][x].y = h_substitutionMatrix[rightY * Dim + rightX];
            }else{
                submat16x2_SubjectSquaredQuerySquared.data[y][x].y = 0; //padding for local alignment
            }
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQuerySquared(), 
        &submat16x2_SubjectSquaredQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear submat16x2_SubjectSquaredQueryLinear;

    for(int y = 0; y < (Dim+1)*(Dim+1); y++){
        const int leftY = y / (Dim+1);
        const int rightY = y % (Dim+1);
        for(int x = 0; x < (Dim+1); x++){
            if(leftY < Dim && x < Dim){
                submat16x2_SubjectSquaredQueryLinear.data[y][x].x = h_substitutionMatrix[leftY * Dim + x];
            }else{
                submat16x2_SubjectSquaredQueryLinear.data[y][x].x = 0; //padding for local alignment
            }
            if(rightY < Dim && x < Dim){
                submat16x2_SubjectSquaredQueryLinear.data[y][x].y = h_substitutionMatrix[rightY * Dim + x];
            }else{
                submat16x2_SubjectSquaredQueryLinear.data[y][x].y = 0; //padding for local alignment
            }
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQueryLinear(), 
        &submat16x2_SubjectSquaredQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared submat16x2_SubjectLinearQuerySquared;

    for(int y = 0; y < (Dim+1); y++){
        for(int x = 0; x < (Dim+1)*(Dim+1); x++){
            const int leftX = x / (Dim+1);
            const int rightX = x % (Dim+1);
            if(y < Dim && leftX < Dim){
                submat16x2_SubjectLinearQuerySquared.data[y][x].x = h_substitutionMatrix[y * Dim + leftX];
            }else{
                submat16x2_SubjectLinearQuerySquared.data[y][x].x = 0; //padding for local alignment
            }
            if(y < Dim && rightX < Dim){
                submat16x2_SubjectLinearQuerySquared.data[y][x].y = h_substitutionMatrix[y * Dim + rightX];
            }else{
                submat16x2_SubjectLinearQuerySquared.data[y][x].y = 0; //padding for local alignment
            }
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQuerySquared(), 
        &submat16x2_SubjectLinearQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear submat16x2_SubjectLinearQueryLinear;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            submat16x2_SubjectLinearQueryLinear.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }
    //padding for local alignment
    for(int x = 0; x < Dim+1; x++){
        submat16x2_SubjectLinearQueryLinear.data[Dim][x] = 0;
    }
    for(int y = 0; y < Dim+1; y++){
        submat16x2_SubjectLinearQueryLinear.data[y][Dim] = 0;
    }
    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(), 
        &submat16x2_SubjectLinearQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));
    

    return gpuSubstitutionMatrix;
}

template<class ScoreType, int Dim>
auto makeGpuSubstitutionMatrix_globalalignment(
    const int* h_substitutionMatrix, //Dim * Dim,  [targetLetter * Dim + queryLetter]
    cudaStream_t stream
){
    using ScoreType32 = typename ToScoreType32<ScoreType>::type;
    constexpr AlignmentType alignmentType = AlignmentType::GlobalAlignment;
    using GpuMatrix = GpuSubstitutionMatrix<alignmentType, ScoreType32, Dim>;
    GpuMatrix gpuSubstitutionMatrix;

    typename GpuMatrix::Submat32 substitutionMatrix32;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            substitutionMatrix32.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat32(), 
        &substitutionMatrix32, 
        sizeof(typename GpuMatrix::Submat32),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared submat16x2_SubjectSquaredQuerySquared;

    for(int y = 0; y < (Dim)*(Dim); y++){
        const int leftY = y / (Dim);
        const int rightY = y % (Dim);
        for(int x = 0; x < (Dim)*(Dim); x++){
            const int leftX = x / (Dim);
            const int rightX = x % (Dim);

            submat16x2_SubjectSquaredQuerySquared.data[y][x].x = h_substitutionMatrix[leftY * Dim + leftX];
            submat16x2_SubjectSquaredQuerySquared.data[y][x].y = h_substitutionMatrix[rightY * Dim + rightX];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQuerySquared(), 
        &submat16x2_SubjectSquaredQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear submat16x2_SubjectSquaredQueryLinear;

    for(int y = 0; y < (Dim)*(Dim); y++){
        const int leftY = y / (Dim);
        const int rightY = y % (Dim);
        for(int x = 0; x < (Dim); x++){
            submat16x2_SubjectSquaredQueryLinear.data[y][x].x = h_substitutionMatrix[leftY * Dim + x];
            submat16x2_SubjectSquaredQueryLinear.data[y][x].y = h_substitutionMatrix[rightY * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQueryLinear(), 
        &submat16x2_SubjectSquaredQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared submat16x2_SubjectLinearQuerySquared;

    for(int y = 0; y < (Dim); y++){
        for(int x = 0; x < (Dim)*(Dim); x++){
            const int leftX = x / (Dim);
            const int rightX = x % (Dim);
            submat16x2_SubjectLinearQuerySquared.data[y][x].x = h_substitutionMatrix[y * Dim + leftX];
            submat16x2_SubjectLinearQuerySquared.data[y][x].y = h_substitutionMatrix[y * Dim + rightX];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQuerySquared(), 
        &submat16x2_SubjectLinearQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear submat16x2_SubjectLinearQueryLinear;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            submat16x2_SubjectLinearQueryLinear.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(), 
        &submat16x2_SubjectLinearQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));
    

    return gpuSubstitutionMatrix;
}

template<class ScoreType, int Dim>
auto makeGpuSubstitutionMatrix_semiglobalalignment(
    const int* h_substitutionMatrix, //Dim * Dim,  [targetLetter * Dim + queryLetter]
    cudaStream_t stream
){
    using ScoreType32 = typename ToScoreType32<ScoreType>::type;
    constexpr AlignmentType alignmentType = AlignmentType::SemiglobalAlignment;
    using GpuMatrix = GpuSubstitutionMatrix<alignmentType, ScoreType32, Dim>;
    GpuMatrix gpuSubstitutionMatrix;

    typename GpuMatrix::Submat32 substitutionMatrix32;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            substitutionMatrix32.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat32(), 
        &substitutionMatrix32, 
        sizeof(typename GpuMatrix::Submat32),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared submat16x2_SubjectSquaredQuerySquared;

    for(int y = 0; y < (Dim)*(Dim); y++){
        const int leftY = y / (Dim);
        const int rightY = y % (Dim);
        for(int x = 0; x < (Dim)*(Dim); x++){
            const int leftX = x / (Dim);
            const int rightX = x % (Dim);

            submat16x2_SubjectSquaredQuerySquared.data[y][x].x = h_substitutionMatrix[leftY * Dim + leftX];
            submat16x2_SubjectSquaredQuerySquared.data[y][x].y = h_substitutionMatrix[rightY * Dim + rightX];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQuerySquared(), 
        &submat16x2_SubjectSquaredQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));

    typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear submat16x2_SubjectSquaredQueryLinear;

    for(int y = 0; y < (Dim)*(Dim); y++){
        const int leftY = y / (Dim);
        const int rightY = y % (Dim);
        for(int x = 0; x < (Dim); x++){
            submat16x2_SubjectSquaredQueryLinear.data[y][x].x = h_substitutionMatrix[leftY * Dim + x];
            submat16x2_SubjectSquaredQueryLinear.data[y][x].y = h_substitutionMatrix[rightY * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectSquaredQueryLinear(), 
        &submat16x2_SubjectSquaredQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectSquaredQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared submat16x2_SubjectLinearQuerySquared;

    for(int y = 0; y < (Dim); y++){
        for(int x = 0; x < (Dim)*(Dim); x++){
            const int leftX = x / (Dim);
            const int rightX = x % (Dim);
            submat16x2_SubjectLinearQuerySquared.data[y][x].x = h_substitutionMatrix[y * Dim + leftX];
            submat16x2_SubjectLinearQuerySquared.data[y][x].y = h_substitutionMatrix[y * Dim + rightX];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQuerySquared(), 
        &submat16x2_SubjectLinearQuerySquared, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQuerySquared),
        cudaMemcpyHostToDevice,
        stream
    ));


    typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear submat16x2_SubjectLinearQueryLinear;
    for(int y = 0; y < Dim; y++){
        for(int x = 0; x < Dim; x++){
            submat16x2_SubjectLinearQueryLinear.data[y][x] = h_substitutionMatrix[y * Dim + x];
        }
    }

    CUDACHECK(cudaMemcpyAsync(
        gpuSubstitutionMatrix.getSubmat16x2_SubjectLinearQueryLinear(), 
        &submat16x2_SubjectLinearQueryLinear, 
        sizeof(typename GpuMatrix::Submat16x2_SubjectLinearQueryLinear),
        cudaMemcpyHostToDevice,
        stream
    ));
    

    return gpuSubstitutionMatrix;
}

template<AlignmentType alignmentType, class ScoreType, int Dim>
auto makeGpuSubstitutionMatrix(
    const int* h_substitutionMatrix, //Dim * Dim,  [targetLetter * Dim + queryLetter]
    cudaStream_t stream
){
    if constexpr(alignmentType == AlignmentType::LocalAlignment){
        return makeGpuSubstitutionMatrix_localalignment<ScoreType, Dim>(h_substitutionMatrix, stream);
    }else if constexpr(alignmentType == AlignmentType::GlobalAlignment){
        return makeGpuSubstitutionMatrix_globalalignment<ScoreType, Dim>(h_substitutionMatrix, stream);
    }else if constexpr(alignmentType == AlignmentType::SemiglobalAlignment){
        return makeGpuSubstitutionMatrix_semiglobalalignment<ScoreType, Dim>(h_substitutionMatrix, stream);
    }else{
        return; //none
    }
}





#endif // SUBSTITUTION_MATRIX_CUH