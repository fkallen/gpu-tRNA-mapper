#ifndef COMMON_CUH
#define COMMON_CUH

#include <string>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <limits>

#include "cuda_errorcheck.cuh"
#include <gpu_api/util.cuh>



#ifdef USE_INTEGER_SCORES
    using ScoreType32 = int;
    using ScoreType16x2 = short2;
#else
    using ScoreType32 = float;
    using ScoreType16x2 = half2;
#endif



struct OneToAllInputDataPSSM{
public: //Our API expects the following declarations
    static constexpr bool isSameQueryForAll = true;

    __host__ __device__
    int getNumAlignments() const{
        return numAlignments;
    }

    __device__
    const std::int8_t* getSubject(int i) const{
        return d_subjects + d_subjectOffsets[i];
    }

    __device__
    int getSubjectLength(int i) const{
        return d_subjectLengths[i];
    }

    __device__
    int getQueryLength(int /*i*/) const{
        return queryLength;
    }

    __host__
    int getMaximumSubjectLength() const{
        return maximumSubjectLength;
    }

public: // API does not care about those
    const std::int8_t* d_subjects;
    const size_t* d_subjectOffsets;
    const int* d_subjectLengths;
    int queryLength;
    int numAlignments;
    int maximumSubjectLength;
};


struct AllToAllInputDataForSubstmat{
public: //Our API expects the following declarations
    static constexpr bool isSameQueryForAll = false;

    __host__ __device__
    int getNumAlignments() const{
        return numSubjects * numQueries;
    }

    __device__
    const std::int8_t* getSubject(int alignmentId) const{
        const int subjectIndex = getSubjectIndex(alignmentId);
        return d_subjects + d_subjectOffsets[subjectIndex];
    }

    __device__
    int getSubjectLength(int alignmentId) const{
        const int subjectIndex = getSubjectIndex(alignmentId);
        return d_subjectLengths[subjectIndex];
    }

    __device__
    const std::int8_t* getQuery(int alignmentId) const{
        const int queryIndex = getQueryIndex(alignmentId);
        return d_queries + d_queryOffsets[queryIndex];
    }

    __device__
    int getQueryLength(int alignmentId) const{
        const int queryIndex = getQueryIndex(alignmentId);
        return d_queryLengths[queryIndex];
    }

    __host__
    int getMaximumSubjectLength() const{
        return maximumSubjectLength;
    }

    __host__
    int getMaximumQueryLength() const{
        return maximumQueryLength;
    }

public: // API does not care about those
    // __host__ __device__
    // int getSubjectIndex(int i) const{
    //     return i / numQueries;
    // }
    // __host__ __device__
    // int getQueryIndex(int i) const{
    //     return i % numQueries;
    // }

    __host__ __device__
    int getSubjectIndex(int i) const{
        return i % numSubjects;
    }
    __host__ __device__
    int getQueryIndex(int i) const{
        return i / numSubjects;
    }

    int numSubjects;
    int numQueries;
    const std::int8_t* d_subjects{};
    const size_t* d_subjectOffsets{};
    const int* d_subjectLengths{};
    const std::int8_t* d_queries{};
    const size_t* d_queryOffsets{};
    const int* d_queryLengths{};
    int maximumSubjectLength{};
    int maximumQueryLength{};
};


struct StartEndPosOfBestInputDataForSubstmat{
public: //Our API expects the following declarations
    static constexpr bool isSameQueryForAll = false;

    __host__ __device__
    int getNumAlignments() const{
        return numAlignments;
    }

    __device__
    const std::int8_t* getSubject(int alignmentId) const{
        const int subjectIndex = getSubjectIndex(alignmentId);
        return d_subjects + d_subjectOffsets[subjectIndex];
    }

    __device__
    int getSubjectLength(int alignmentId) const{
        const int subjectIndex = getSubjectIndex(alignmentId);
        return d_subjectLengths[subjectIndex];
    }

    __device__
    const std::int8_t* getQuery(int alignmentId) const{
        const int queryIndex = getQueryIndex(alignmentId);
        return d_queries + d_queryOffsets[queryIndex];
    }

    __device__
    int getQueryLength(int alignmentId) const{
        const int queryIndex = getQueryIndex(alignmentId);
        return d_queryLengths[queryIndex];
    }

    __host__
    int getMaximumSubjectLength() const{
        return maximumSubjectLength;
    }

    __host__
    int getMaximumQueryLength() const{
        return maximumQueryLength;
    }

public: // API does not care about those
    __host__ __device__
    int getSubjectIndex(int alignmentId) const{
        return d_segmentIdsPerElement[alignmentId];
    }

    __host__ __device__
    int getQueryIndex(int alignmentId) const{
        return d_referenceSequenceIdsOfMax[alignmentId];
    }

    int numSubjects;
    int numQueries;
    const std::int8_t* d_subjects{};
    const size_t* d_subjectOffsets{};
    const int* d_subjectLengths{};
    const std::int8_t* d_queries{};
    const size_t* d_queryOffsets{};
    const int* d_queryLengths{};
    int maximumSubjectLength{};
    int maximumQueryLength{};
    int numAlignments{};
    const int* d_segmentIdsPerElement{};
    const int* d_referenceSequenceIdsOfMax{};
    const int* d_numReferenceSequenceIdsPerReadPrefixSum{};
};



template <class T>
struct PinnedAllocator {
    using value_type = T;

    // PinnedAllocator() = default;

    // template <class U>
    // constexpr Mallocator(const Mallocator<U>&) noexcept {}

    T* allocate(size_t elements){
        T* ptr{};
        cudaError_t err = cudaMallocHost(&ptr, elements * sizeof(T));
        if(err != cudaSuccess){
            std::cerr << "SimpleAllocator: Failed to allocate " << (elements) << " * " << sizeof(T)
                        << " = " << (elements * sizeof(T))
                        << " bytes using cudaMallocHost!\n";

            throw std::bad_alloc();
        }

        assert(ptr != nullptr);

        return ptr;
    }

    void deallocate(T* ptr, size_t /*n*/){
        CUDACHECK(cudaFreeHost(ptr));
    }
};

template<class T>
struct SimpleConcurrentQueue{
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;

    void push(T item){
        std::lock_guard<std::mutex> lg(mutex);
        queue.emplace(std::move(item));
        cv.notify_one();
    }

    //wait until queue is not empty, then remove first element from queue and return it
    T pop(){
        std::unique_lock<std::mutex> ul(mutex);

        while(queue.empty()){
            cv.wait(ul);
        }

        T item = queue.front();
        queue.pop();
        return item;
    }
};



struct Options{
    bool verbose = false;
    bool use16x2 = false;
    int batchsize = 100'000;
    int minAlignmentScore = 0;
    int queue_depth = 4;
    int numBatches = 0;
    int resultListSize = std::numeric_limits<int>::max();
    AlignmentType alignmentType = AlignmentType::LocalAlignment;
    std::string readFileName = "";
    std::string referenceFileName = "";
    std::string outputFileName = "";
    Scoring1 scoring{};
};



struct SequenceCollection{
    bool hasQualityScores = false;
    size_t sumOfLengths = 0;
    int maximumSequenceLength = 0;
    std::vector<char, PinnedAllocator<char>> h_sequences{};
    std::vector<int, PinnedAllocator<int>> h_lengths{};
    std::vector<size_t, PinnedAllocator<size_t>> h_offsets{};
    std::vector<std::string> headers{};
    std::vector<std::string> sequenceNames{};
    std::vector<char> qualities{};

    void clear(){
        hasQualityScores = false;
        sumOfLengths = 0;
        maximumSequenceLength = 0;
        h_sequences.clear();
        h_lengths.clear();
        h_offsets.clear();
        headers.clear();
        sequenceNames.clear();
        qualities.clear();
    }
};


#endif