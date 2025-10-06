#ifndef LENGTH_SWITCHER_API_CUH
#define LENGTH_SWITCHER_API_CUH

#include "util.cuh"
#include "substitutionmatrix.cuh"

#include <array>
#include <algorithm>



template<int blocksize_, int groupsize_, int numItems_>
struct KernelConfig{
    static constexpr int blocksize = blocksize_;
    static constexpr int groupsize = groupsize_;
    static constexpr int numItems = numItems_;
    static constexpr int tileSize = groupsize * numItems;
    static constexpr int groupsPerBlock = blocksize / groupsize;
};

template <
    template<int,class,PenaltyType,int,int,int> class Aligner, 
    int alphabetSize, 
    class ScoreType, 
    PenaltyType penaltyType, 
    class KernelConfig
> 
struct AlignerFromKernelConfig;

template<
    template<int,class,PenaltyType,int,int,int> class Aligner, 
    int alphabetSize, 
    class ScoreType, 
    PenaltyType penaltyType, 
    int blocksize, 
    int groupsize, 
    int numItems
>
struct AlignerFromKernelConfig<Aligner, alphabetSize, ScoreType, penaltyType, KernelConfig<blocksize, groupsize, numItems>>
{
    using type = Aligner<alphabetSize, ScoreType, penaltyType, blocksize, groupsize, numItems>;
};

template <
    template<int,class,PenaltyType,int,int,int> class Aligner, 
    int alphabetSize, 
    class ScoreType, 
    PenaltyType penaltyType, 
    class Tuple, 
    template<
        template<int,class,PenaltyType,int,int,int> class, 
        int, 
        class, 
        PenaltyType, 
        class
    > class Transformer
>
struct ConfiguredAlignersFromTuple;

template <
    template<int,class,PenaltyType,int,int,int> class Aligner, 
    int alphabetSize, 
    class ScoreType, 
    PenaltyType penaltyType, 
    class... Args, 
    template<
        template<int,class,PenaltyType,int,int,int> class, 
        int, 
        class, 
        PenaltyType, 
        class
    > class Transformer
>
struct ConfiguredAlignersFromTuple<Aligner, alphabetSize, ScoreType, penaltyType, std::tuple<Args...>, Transformer>
{
    using type = std::tuple<class Transformer<Aligner, alphabetSize, ScoreType, penaltyType, Args>::type...>;
};

template<class ConfigTuple>
struct ArrayOfTileSizes;

template<class... Configs>
struct ArrayOfTileSizes<std::tuple<Configs...>>{
    static constexpr std::array<int, sizeof...(Configs)> array{Configs::tileSize ...};
};

template<class ConfigTuple>
struct ArrayOfNumItems;

template<class... Configs>
struct ArrayOfNumItems<std::tuple<Configs...>>{
    static constexpr std::array<int, sizeof...(Configs)> array{Configs::numItems ...};
};

template<class ConfigTuple>
struct ArrayOfGroupsize;

template<class... Configs>
struct ArrayOfGroupsize<std::tuple<Configs...>>{
    static constexpr std::array<int, sizeof...(Configs)> array{Configs::groupsize ...};
};



template<class ConfigTuple>
struct ArrayOfGroupsPerBlock;

template<class... Configs>
struct ArrayOfGroupsPerBlock<std::tuple<Configs...>>{
    static constexpr std::array<int, sizeof...(Configs)> array{Configs::groupsPerBlock ...};
};


template<class... Types>
struct ListOfTypes{
    using type = typename std::tuple<Types...>;
    static constexpr int size = std::tuple_size_v<type>;
};

template<
    template<int,class,PenaltyType,int,int,int> class Aligner, 
    int alphabetSize_, 
    class ScoreType_, 
    PenaltyType penaltyType_, 
    class ListOfTypesShortQuery,
    class ListOfTypesLongQuery
>
struct GpuAlignerWithLengthSwitch{

    using ScoreType = ScoreType_;
    static constexpr int alphabetSize = alphabetSize_;
    static constexpr PenaltyType penaltyType = penaltyType_;

    using ConfiguredTupleShortQuery = typename ConfiguredAlignersFromTuple<Aligner, alphabetSize, ScoreType, penaltyType, typename ListOfTypesShortQuery::type, AlignerFromKernelConfig>::type;
    using ConfiguredTupleLongQuery = typename ConfiguredAlignersFromTuple<Aligner, alphabetSize, ScoreType, penaltyType, typename ListOfTypesLongQuery::type, AlignerFromKernelConfig>::type;

    static_assert(std::tuple_size_v<ConfiguredTupleShortQuery> + std::tuple_size_v<ConfiguredTupleLongQuery> > 0);

    static constexpr AlignmentType alignmentType = [](){
        if constexpr(std::tuple_size_v<ConfiguredTupleShortQuery> > 0){
            using CurrentAligner = typename std::tuple_element<0, ConfiguredTupleShortQuery>::type;
            return CurrentAligner::alignmentType;
        }else{
            using CurrentAligner = typename std::tuple_element<0, ConfiguredTupleLongQuery>::type;
            return CurrentAligner::alignmentType;
        }
    }();

    static constexpr int alphabetSizeWithPadding = [](){
        if constexpr(alignmentType == AlignmentType::LocalAlignment){
            return alphabetSize+1;
        }else{
            return alphabetSize;
        }
        // if constexpr(std::tuple_size_v<ConfiguredTupleShortQuery> > 0){
        //     using CurrentAligner = typename std::tuple_element<0, ConfiguredTupleShortQuery>::type;
        //     return CurrentAligner::alphabetSizeWithPadding;
        // }else{
        //     using CurrentAligner = typename std::tuple_element<0, ConfiguredTupleLongQuery>::type;
        //     return CurrentAligner::alphabetSizeWithPadding;
        // }
    }();

    static constexpr int largestShortQueryTileSize = [](){
        if constexpr(std::tuple_size_v<ConfiguredTupleShortQuery> > 0){
            using CurrentAligner = typename std::tuple_element<
                std::tuple_size_v<ConfiguredTupleShortQuery> - 1, ConfiguredTupleShortQuery>::type;
            return CurrentAligner::groupsize * CurrentAligner::numItems;
        }else{
            return 0;
        }
    }();


    using GpuSubstitutionMatrixType = GpuSubstitutionMatrix<alignmentType, typename ToScoreType32<ScoreType>::type, alphabetSizeWithPadding>;

    ConfiguredTupleShortQuery shortQueryAligners;
    ConfiguredTupleLongQuery longQueryAligners;


    size_t getMinimumSuggestedTempBytes_longQuery(int maximumSubjectLength, int maximumQueryLength) const{

        // auto getTempBytesPerAligner = [&](const auto& ... aligner){ 
        //     return std::array{aligner.getMinimumSuggestedTempBytes_longQuery(maximumSubjectLength) ... }; 
        // };
        // const auto array = std::apply(getTempBytesPerAligner, longQueryAligners);
        // if(array.size() > 0){
        //     return *std::max_element(array.begin(), array.end());
        // }else{
        //     throw std::runtime_error("longQuery alignment requires kernel config for long queries");
        // }

        const int selectedConfigIndex = getLongQueryConfigIndex(maximumQueryLength);
        return getMinimumSuggestedTempBytes_longQuery_helper<0>(selectedConfigIndex, maximumSubjectLength);
    }



    template<
        class SequenceInputDataOneToOne
    >
    void oneToOne_shortQuery(
        int maximumQueryLength,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& encodedInputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        oneToOne_shortQuery_helper<0>(
            maximumQueryLength,
            d_scoreOutput,
            gpuSubstitutionMatrix,
            encodedInputData,
            scoring,
            stream
        );
    }


    template<
        class SequenceInputDataOneToOne
    >
    void oneToOne_longQuery(
        int maximumQueryLength,
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& encodedInputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        const int selectedConfigIndex = getLongQueryConfigIndex(maximumQueryLength);
        oneToOne_longQuery_helper<0>(
            selectedConfigIndex,
            maximumQueryLength,
            d_temp,
            tempBytes,
            d_scoreOutput,
            gpuSubstitutionMatrix,
            encodedInputData,
            scoring,
            stream
        );

    }

private:
    int getLongQueryConfigIndex(int maximumQueryLength) const{
        constexpr auto array = ArrayOfTileSizes<typename ListOfTypesLongQuery::type>::array;
        if constexpr(array.size() > 0){
            #if 0
                //find the config which best utilizes the last tile. larger tile sizes are preferred
                int selectedConfig = 0;
                const int remainderInLastTile0 = maximumQueryLength % array[0];
                double utilization = remainderInLastTile0 == 0 ? 1.0 : double(remainderInLastTile0) / array[0];
                for(size_t i = 1; i < array.size(); i++){
                    const int remainderInLastTile = maximumQueryLength % array[i];
                    const double newUtilization = remainderInLastTile == 0 ? 1.0 : double(remainderInLastTile) / array[i];
                    if(newUtilization >= utilization){
                        utilization = newUtilization;
                        selectedConfig = i;
                    }
                }
            #else

                //determine minimum number of tiles
                std::vector<int> numTilesPerConfig(array.size());
                for(size_t i = 0; i < array.size(); i++){
                    numTilesPerConfig[i] = SDIV(maximumQueryLength, array[i]);
                }
                const int minTiles = *std::min_element(numTilesPerConfig.begin(), numTilesPerConfig.end());

                //from those configs which require minimum number of tiles, use the config with best utilization in last tile
                int selectedConfig = -1;
                // const int remainderInLastTile0 = 99999;
                double utilization = 0;
                for(size_t i = 0; i < array.size(); i++){
                    const auto& newConfig = array[i];
                    if(numTilesPerConfig[i] == minTiles){
                        const int remainderInLastTile = maximumQueryLength % array[i];
                        const double newUtilization = remainderInLastTile == 0 ? 1.0 : double(remainderInLastTile) / array[i];
                        if(newUtilization >= utilization){
                            utilization = newUtilization;
                            selectedConfig = i;
                        }
                    }
                }
            #endif            

            return selectedConfig;
        }else{
            throw std::runtime_error("longQuery alignment requires kernel config for long queries");
        }
    }

    int getLongQueryConfigIndex(int groupsize, int numItems) const{
        constexpr auto numItemsArray = ArrayOfNumItems<typename ListOfTypesLongQuery::type>::array;
        constexpr auto groupsizeArray = ArrayOfGroupsize<typename ListOfTypesLongQuery::type>::array;

        for(size_t i = 0; i < numItemsArray.size(); i++){
            if(groupsize == groupsizeArray[i] && numItems == numItemsArray[i]){
                return i;
            }
        }
        throw std::runtime_error("getLongQueryConfigIndex: no config with groupsize/numItems");
    }

    template<int nr>
    size_t getMinimumSuggestedTempBytes_longQuery_helper(int selectedConfigIndex, int maximumSubjectLength) const{
        if constexpr(std::tuple_size_v<ConfiguredTupleLongQuery> > 0){
            using CurrentAligner = typename std::tuple_element<nr, ConfiguredTupleLongQuery>::type;
            auto& current = std::get<nr>(longQueryAligners);

            if(selectedConfigIndex == nr){
                // std::cout << "using config " << CurrentAligner::groupsize << " * " << CurrentAligner::numItems << "\n";
                return current.getMinimumSuggestedTempBytes_longQuery(maximumSubjectLength);
            }else{
                if constexpr(nr+1 < std::tuple_size_v<ConfiguredTupleLongQuery>){
                    return getMinimumSuggestedTempBytes_longQuery_helper<nr+1>(
                        selectedConfigIndex,
                        maximumSubjectLength
                    );
                }else{
                    throw std::runtime_error("error longQuery switch invalid config");
                }
            }
        }else{
            return 0;
        }

    }



    template<
        int nr,
        class SequenceInputDataOneToAll
    >
    void oneToAll_shortQuery_helper(
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        if constexpr(std::tuple_size_v<ConfiguredTupleShortQuery> > 0){
            using CurrentAligner = typename std::tuple_element<nr, ConfiguredTupleShortQuery>::type;
            auto& current = std::get<nr>(shortQueryAligners);

            if(inputData.getQueryLength() <= CurrentAligner::groupsize * CurrentAligner::numItems){
                current.oneToAll_shortQuery(
                    d_scoreOutput,
                    gpuSubstitutionMatrix,
                    inputData,
                    scoring,
                    stream
                );
            }else{
                if constexpr(nr+1 < std::tuple_size_v<ConfiguredTupleShortQuery>){
                    oneToAll_shortQuery_helper<nr+1, SequenceInputDataOneToAll>(
                        d_scoreOutput,
                        gpuSubstitutionMatrix,
                        inputData,
                        scoring,
                        stream
                    );
                }else{
                    throw std::runtime_error("error shortQuery switch length exceeded");
                }
            }
        }else{
            throw std::runtime_error("shortQuery alignment requires kernel config for short queries");
        }
    }

    template<
        int nr,
        class SequenceInputDataOneToAll
    >
    void oneToAll_longQuery_helper(
        int selectedConfigIndex,
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToAll& inputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        if constexpr(std::tuple_size_v<ConfiguredTupleLongQuery> > 0){
            using CurrentAligner = typename std::tuple_element<nr, ConfiguredTupleLongQuery>::type;
            auto& current = std::get<nr>(longQueryAligners);

            if(selectedConfigIndex == nr){
                current.oneToAll_longQuery(
                    d_temp,
                    tempBytes,
                    d_scoreOutput,
                    gpuSubstitutionMatrix,
                    inputData,
                    scoring,
                    stream
                );
            }else{
                if constexpr(nr+1 < std::tuple_size_v<ConfiguredTupleLongQuery>){
                    oneToAll_longQuery_helper<nr+1, SequenceInputDataOneToAll>(
                        selectedConfigIndex,
                        d_temp,
                        tempBytes,
                        d_scoreOutput,
                        gpuSubstitutionMatrix,
                        inputData,
                        scoring,
                        stream
                    );
                }else{
                    throw std::runtime_error("error longQuery switch invalid config");
                }
            }
        }else{
            throw std::runtime_error("longQuery alignment requires kernel config for long queries");
        }
    }



    template<
        int nr,
        class SequenceInputDataOneToOne
    >
    void oneToOne_shortQuery_helper(
        int maximumQueryLength,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& encodedInputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        if constexpr(std::tuple_size_v<ConfiguredTupleShortQuery> > 0){
            using CurrentAligner = typename std::tuple_element<nr, ConfiguredTupleShortQuery>::type;
            auto& current = std::get<nr>(shortQueryAligners);

            if(maximumQueryLength <= CurrentAligner::groupsize * CurrentAligner::numItems){
                current.oneToOne_shortQuery(
                    d_scoreOutput,
                    gpuSubstitutionMatrix,
                    encodedInputData,
                    scoring,
                    stream
                );
            }else{
                if constexpr(nr+1 < std::tuple_size_v<ConfiguredTupleShortQuery>){
                    oneToOne_shortQuery_helper<nr+1>(
                        maximumQueryLength,
                        d_scoreOutput,
                        gpuSubstitutionMatrix,
                        encodedInputData,
                        scoring,
                        stream
                    );
                }else{
                    throw std::runtime_error("error shortQuery switch length exceeded");
                }
            }
        }else{
            throw std::runtime_error("shortQuery alignment requires kernel config for short queries");
        }
    }

    template<
        int nr,
        class SequenceInputDataOneToOne
    >
    void oneToOne_longQuery_helper(
        int selectedConfigIndex,
        int maximumQueryLength,
        char* d_temp,
        size_t tempBytes,
        int* d_scoreOutput,
        const GpuSubstitutionMatrixType& gpuSubstitutionMatrix,
        const SequenceInputDataOneToOne& encodedInputData,
        Scoring1 scoring,
        cudaStream_t stream
    ){
        if constexpr(std::tuple_size_v<ConfiguredTupleLongQuery> > 0){
            using CurrentAligner = typename std::tuple_element<nr, ConfiguredTupleLongQuery>::type;
            auto& current = std::get<nr>(longQueryAligners);

            if(selectedConfigIndex == nr){
                current.oneToOne_longQuery(
                    d_temp,
                    tempBytes,
                    d_scoreOutput,
                    gpuSubstitutionMatrix,
                    encodedInputData,
                    scoring,
                    stream
                );
            }else{
                if constexpr(nr+1 < std::tuple_size_v<ConfiguredTupleLongQuery>){
                    oneToOne_longQuery_helper<nr+1>(
                        selectedConfigIndex,
                        maximumQueryLength,
                        d_temp,
                        tempBytes,
                        d_scoreOutput,
                        gpuSubstitutionMatrix,
                        encodedInputData,
                        scoring,
                        stream
                    );
                }else{
                    throw std::runtime_error("error longQuery switch invalid config");
                }
            }
        }else{
            throw std::runtime_error("longQuery alignment requires kernel config for long queries");
        }

    }


};







#endif