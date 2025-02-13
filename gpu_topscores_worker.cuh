#ifndef GPU_TOPSCORES_WORKER_CUH
#define GPU_TOPSCORES_WORKER_CUH

#include "execution_pipeline.cuh"

#include "common.cuh"
#include "alphabet.cuh"
#include "smallkernels.cuh"

#include <rmm/exec_policy.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "raii_cudaevent.cuh"
#include "raii_cudastream.cuh"
#include "hpc_helpers/timers.cuh"

#include <gpu_api/substitutionmatrix.cuh>
#include <gpu_api/semiglobal_alignment/kernels_startendpos.cuh>
#include <gpu_api/local_alignment/kernels_startendpos.cuh>



template<
    int alphabetSize,
    class ScoreType, 
    PenaltyType penaltyType, 
    int blocksize, 
    int groupsize, 
    int numItems,
    class InputData,
    class SUBMAT
>
void computeAlignmentsStartAndEndpositions_semiglobalAlignment_floatOrInt_multitile(
    int* d_scoreOutput,
    int* d_queryEndPositions_inclusive,
    int* d_subjectEndPositions_inclusive,
    int* d_queryStartPositions_inclusive,
    int* d_subjectStartPositions_inclusive,
    const InputData& inputData,
    const SUBMAT* d_substmatPtr,
    const ScoringKernelParam<ScoreType>& scoring,
    int maxSubjectLength,
    char* d_temp, //must be aligned to 256 bytes
    size_t tempBytes,
    cudaStream_t stream
){
    semiglobalalignment::call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel<alphabetSize,ScoreType,penaltyType,blocksize,groupsize,numItems>(
        d_scoreOutput,
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
        inputData,
        d_substmatPtr,
        scoring,
        maxSubjectLength,
        d_temp,
        tempBytes,
        stream
    );

    semiglobalalignment::call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_backwardpass_kernel<alphabetSize,ScoreType,penaltyType,blocksize,groupsize,numItems>(
        d_queryStartPositions_inclusive,
        d_subjectStartPositions_inclusive,
        d_scoreOutput,
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
        inputData,
        d_substmatPtr,
        scoring,
        maxSubjectLength,
        d_temp,
        tempBytes,
        stream
    );
}

template<
    int alphabetSize,
    class ScoreType, 
    PenaltyType penaltyType, 
    int blocksize, 
    int groupsize, 
    int numItems,
    class InputData,
    class SUBMAT
>
void computeAlignmentsStartAndEndpositions_localAlignment_floatOrInt_multitile(
    int* d_scoreOutput,
    int* d_queryEndPositions_inclusive,
    int* d_subjectEndPositions_inclusive,
    int* d_queryStartPositions_inclusive,
    int* d_subjectStartPositions_inclusive,
    const InputData& inputData,
    const SUBMAT* d_substmatPtr,
    const ScoringKernelParam<ScoreType>& scoring,
    int maxSubjectLength,
    char* d_temp, //must be aligned to 256 bytes
    size_t tempBytes,
    cudaStream_t stream
){
    localalignment::call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_forwardpass_kernel<alphabetSize,ScoreType,penaltyType,blocksize,groupsize,numItems>(
        d_scoreOutput,
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
        inputData,
        d_substmatPtr,
        scoring,
        maxSubjectLength,
        d_temp,
        tempBytes,
        stream
    );

    localalignment::call_alphabet_substitutionmatrix_floatOrInt_multitile_withStartAndEndPos_backwardpass_kernel<alphabetSize,ScoreType,penaltyType,blocksize,groupsize,numItems>(
        d_queryStartPositions_inclusive,
        d_subjectStartPositions_inclusive,
        d_scoreOutput,
        d_queryEndPositions_inclusive,
        d_subjectEndPositions_inclusive,
        inputData,
        d_substmatPtr,
        scoring,
        maxSubjectLength,
        d_temp,
        tempBytes,
        stream
    );
}




template<class GpuAlignerSwitch_>
struct GpuTopScoresAlignerWorker{
    using GpuAligner = GpuAlignerSwitch_;

    const Options* optionsPtr;
    BatchDataQueue* batchDataQueueIn;
    BatchDataQueue* batchDataQueueOut;

    GpuAligner gpuAligner{};
    std::vector<std::vector<int>> substitutionMatrix2D;
    typename GpuAligner::GpuSubstitutionMatrixType gpuSubstitutionMatrix;
    parasail_matrix_t* parasail_scoring_matrix;

    GpuTopScoresAlignerWorker(
        const Options* optionsPtr_, 
        const std::vector<std::vector<int>>& substitutionMatrix2D_,
        parasail_matrix_t* parasail_scoring_matrix_,
        BatchDataQueue* batchDataQueueIn_, 
        BatchDataQueue* batchDataQueueOut_
    )
        : optionsPtr(optionsPtr_), 
        batchDataQueueIn(batchDataQueueIn_), 
        batchDataQueueOut(batchDataQueueOut_),
        substitutionMatrix2D(substitutionMatrix2D_),
        gpuSubstitutionMatrix([&](){
            std::vector<int> substitutionMatrix1D(alphabetSize * alphabetSize);
            for(int r = 0; r < alphabetSize; r++){
                for(int c = 0; c < alphabetSize; c++){
                    substitutionMatrix1D[r * alphabetSize + c] = substitutionMatrix2D[r][c];
                }
            }
        
            auto res = makeGpuSubstitutionMatrix<GpuAligner::alignmentType, typename GpuAligner::ScoreType, GpuAligner::alphabetSize>(
                substitutionMatrix1D.data(),
                (cudaStream_t)0
            );
            CUDACHECK(cudaDeviceSynchronize());
            return res;
        }()),
        parasail_scoring_matrix(parasail_scoring_matrix_)
    {
    }

    void run(){
        nvtx3::scoped_range sr("GpuTopScoresAlignerWorker::run");
        helpers::CpuTimer cputimer("GpuTopScoresAlignerWorker::run");
        const auto& options = *optionsPtr;
        std::vector<CudaStream> streams(4);
        std::vector<cudaStream_t> rawstreams;
        for(auto& stream : streams){
            rawstreams.push_back(stream.getStream());
        }

        BatchData* batchDataPtr = batchDataQueueIn->pop();

        while(batchDataPtr != nullptr){
            helpers::CpuTimer timer1("batch " + std::to_string(batchDataPtr->batchId) + " computeAllToAllScoresGpu");
            computeAllToAllScoresGpu(
                batchDataPtr,
                rawstreams
            );
            CUDACHECK(cudaDeviceSynchronize());
            if(options.verbose){
                timer1.print();
            }
            batchDataPtr->computedCells = batchDataPtr->readSequences.sumOfLengths * batchDataPtr->referenceSequences->sumOfLengths;

            batchDataQueueOut->push(batchDataPtr);
            batchDataPtr = batchDataQueueIn->pop();
        }

        //notify writer
        batchDataQueueOut->push(nullptr);

        if(options.verbose){
            cputimer.print();
        }
    }

    void computeAllToAllScoresGpu(
        BatchData* batchDataPtr,
        std::vector<cudaStream_t>& streams
    ){
        static_assert(sizeof(char) == sizeof(std::int8_t));
        nvtx3::scoped_range sr("computeAllToAllScoresGpu");

        const auto& options = *optionsPtr;
    
        const SequenceCollection& readSequences = batchDataPtr->readSequences;
        const SequenceCollection& referenceSequences = *batchDataPtr->referenceSequences;
    
        const int numReadSequences = readSequences.h_lengths.size();
        const int numReferenceSequences = referenceSequences.h_lengths.size();
        
        auto& mainStream = streams[0];
        auto& stream2 = streams[(1) % streams.size()];   
    
        rmm::device_uvector<std::int8_t> d_read_sequences(readSequences.h_sequences.size(), mainStream);
        rmm::device_uvector<int> d_read_lengths(readSequences.h_lengths.size(), mainStream);
        rmm::device_uvector<size_t> d_read_offsets(readSequences.h_offsets.size(), mainStream);
        rmm::device_uvector<std::int8_t> d_reference_sequences(referenceSequences.h_sequences.size(), mainStream);
        rmm::device_uvector<int> d_reference_lengths(referenceSequences.h_lengths.size(), mainStream);
        rmm::device_uvector<size_t> d_reference_offsets(referenceSequences.h_offsets.size(), mainStream);
    
        CUDACHECK(cudaDeviceSynchronize()); // wait for allocations
    
    
        CUDACHECK(cudaEventRecord(batchDataPtr->h2d_Start, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_read_sequences.data(), readSequences.h_sequences.data(), sizeof(char) * readSequences.h_sequences.size(), cudaMemcpyHostToDevice, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_reference_sequences.data(), referenceSequences.h_sequences.data(), sizeof(char) * referenceSequences.h_sequences.size(), cudaMemcpyHostToDevice, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_read_lengths.data(), readSequences.h_lengths.data(), sizeof(int) * readSequences.h_lengths.size(), cudaMemcpyHostToDevice, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_read_offsets.data(), readSequences.h_offsets.data(), sizeof(size_t) * readSequences.h_offsets.size(), cudaMemcpyHostToDevice, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_reference_lengths.data(), referenceSequences.h_lengths.data(), sizeof(int) * referenceSequences.h_lengths.size(), cudaMemcpyHostToDevice, mainStream));
        CUDACHECK(cudaMemcpyAsync(d_reference_offsets.data(), referenceSequences.h_offsets.data(), sizeof(size_t) * referenceSequences.h_offsets.size(), cudaMemcpyHostToDevice, mainStream));

        CUDACHECK(cudaEventRecord(batchDataPtr->h2d_Stop, mainStream));

        CUDACHECK(cudaEventRecord(batchDataPtr->otherkernels_Start, mainStream));
        thrust::transform(rmm::exec_policy_nosync(mainStream), d_read_sequences.begin(), d_read_sequences.end(), d_read_sequences.begin(), ConvertLetters_functor{});
        thrust::transform(rmm::exec_policy_nosync(mainStream), d_reference_sequences.begin(), d_reference_sequences.end(), d_reference_sequences.begin(), ConvertLetters_functor{});

        rmm::device_uvector<int> d_originalInputPositions(numReadSequences, mainStream);
        thrust::sequence(rmm::exec_policy_nosync(mainStream), d_originalInputPositions.begin(), d_originalInputPositions.end(), 0);

        //our kernels are more efficient if target lengths are sorted
        thrust::sort_by_key(
            rmm::exec_policy_nosync(mainStream),
            d_read_lengths.begin(),
            d_read_lengths.end(),
            thrust::make_zip_iterator(
                d_read_offsets.begin(),
                d_originalInputPositions.begin()
            )
        );

        CUDACHECK(cudaEventRecord(batchDataPtr->otherkernels_Stop, mainStream));

        AllToAllInputDataForSubstmat inputData;
        inputData.numSubjects = numReadSequences;
        inputData.numQueries = numReferenceSequences;
        inputData.d_subjects = d_read_sequences.data();
        inputData.d_subjectOffsets = d_read_offsets.data();
        inputData.d_subjectLengths = d_read_lengths.data();
        inputData.d_queries = d_reference_sequences.data();
        inputData.d_queryOffsets = d_reference_offsets.data(),
        inputData.d_queryLengths = d_reference_lengths.data();
        inputData.maximumSubjectLength = readSequences.maximumSequenceLength;
        inputData.maximumQueryLength = referenceSequences.maximumSequenceLength;


        rmm::device_uvector<int> d_allScores(numReferenceSequences * numReadSequences, mainStream);
        rmm::device_uvector<char> d_temp(0, mainStream); 

        if(referenceSequences.maximumSequenceLength > GpuAligner::largestShortQueryTileSize){
            size_t bytes = gpuAligner.getMinimumSuggestedTempBytes_longQuery(readSequences.maximumSequenceLength, referenceSequences.maximumSequenceLength);
            d_temp.resize(bytes, mainStream);
        }

        helpers::GpuTimer timerKernelLoop(mainStream, "batch " + std::to_string(batchDataPtr->batchId) + " alignment kernels");
        CUDACHECK(cudaEventRecord(batchDataPtr->alignmentkernels_Start, mainStream));
        if(referenceSequences.maximumSequenceLength <= GpuAligner::largestShortQueryTileSize){
            gpuAligner.oneToOne_shortQuery(
                referenceSequences.maximumSequenceLength,
                d_allScores.data(),
                gpuSubstitutionMatrix,
                inputData,
                options.scoring,
                mainStream
            );
        }else{
            gpuAligner.oneToOne_longQuery(
                referenceSequences.maximumSequenceLength,
                d_temp.data(),
                d_temp.size(),
                d_allScores.data(),
                gpuSubstitutionMatrix,
                inputData,
                options.scoring,
                mainStream
            );
        }
        for(auto& stream : streams){
            CUDACHECK(cudaStreamSynchronize(stream));
        }
        CUDACHECK(cudaEventRecord(batchDataPtr->alignmentkernels_Stop, mainStream));
        timerKernelLoop.stop();
        if(options.verbose){
            timerKernelLoop.printGCUPS(size_t(readSequences.sumOfLengths) * size_t(referenceSequences.sumOfLengths));
        }

        d_temp.release();

        rmm::device_uvector<int> d_maxScores(numReadSequences, mainStream);
        rmm::device_uvector<int> d_numBestResultsPerRead(numReadSequences, mainStream);
        rmm::device_uvector<int> d_numBestResultsPerReadPrefixSum(numReadSequences+1, mainStream);
        d_numBestResultsPerReadPrefixSum.set_element_to_zero_async(0, mainStream);
        // rmm::device_uvector<int> d_referenceSequenceIdOfMaxScore(numReferenceSequences * numReadSequences, mainStream);
        // horizontalMaxReduceWithIndexKernel<<<SDIV(numReadSequences, 128/32), 128, 0, mainStream>>>(
        //     d_allScores.data(), 
        //     numReadSequences, 
        //     numReferenceSequences, 
        //     d_maxScores.data(), 
        //     d_referenceSequenceIdOfMaxScore.data()
        // );

        // verticalMaxReduceWithIndexKernel<<<SDIV(numReadSequences, 128), 128, 0, mainStream>>>(
        //     d_allScores.data(), 
        //     numReferenceSequences, 
        //     numReadSequences, 
        //     d_maxScores.data(), 
        //     d_referenceSequenceIdOfMaxScore.data()
        // );

        callVerticalMaxReduceFindNumBestScoresKernel(
            d_allScores.data(), 
            numReferenceSequences, 
            numReadSequences, 
            d_maxScores.data(), 
            d_numBestResultsPerRead.data(),
            options.minAlignmentScore,
            mainStream
        );

        thrust::inclusive_scan(rmm::exec_policy_nosync(mainStream), d_numBestResultsPerRead.begin(), d_numBestResultsPerRead.end(), d_numBestResultsPerReadPrefixSum.begin() + 1);
        const int totalNumBestResults = d_numBestResultsPerReadPrefixSum.element(numReadSequences, mainStream);
        rmm::device_uvector<int> d_referenceSequenceIdsOfMaxScore(totalNumBestResults, mainStream);
        callVerticalMaxReduceFindIndicesOfBestScoresKernel(
            d_allScores.data(), 
            numReferenceSequences, 
            numReadSequences, 
            d_maxScores.data(), 
            d_numBestResultsPerRead.data(),
            d_numBestResultsPerReadPrefixSum.data(),
            options.minAlignmentScore,
            d_referenceSequenceIdsOfMaxScore.data(),
            mainStream
        );

        // std::cout << "totalNumBestResults " << totalNumBestResults << ", numReadSequences " << numReadSequences << ", ratio " << (totalNumBestResults / float(numReadSequences)) << "\n";
        d_allScores.release();


        rmm::device_uvector<int> d_segmentIdsPerElement = getSegmentIdsPerElement(
            d_numBestResultsPerRead.data(),
            d_numBestResultsPerReadPrefixSum.data(),
            numReadSequences, 
            totalNumBestResults,
            mainStream
        );
        rmm::device_uvector<int> d_queryEndPositions_inclusive(totalNumBestResults, mainStream);
        rmm::device_uvector<int> d_subjectEndPositions_inclusive(totalNumBestResults, mainStream);
        rmm::device_uvector<int> d_queryStartPositions_inclusive(totalNumBestResults, mainStream);
        rmm::device_uvector<int> d_subjectStartPositions_inclusive(totalNumBestResults, mainStream);
        if constexpr(GpuAligner::alignmentType == AlignmentType::LocalAlignment){
            //compute the start and end positions for the best scoring alignments per read

            rmm::device_uvector<int> d_bestScores_startendpos(totalNumBestResults, mainStream);
            d_temp.resize(size_t(1) << 31, mainStream);

            StartEndPosOfBestInputDataForSubstmat startendposInput;
            startendposInput.numSubjects = numReadSequences;
            startendposInput.numQueries = numReferenceSequences;
            startendposInput.d_subjects = d_read_sequences.data();
            startendposInput.d_subjectOffsets = d_read_offsets.data();
            startendposInput.d_subjectLengths = d_read_lengths.data();
            startendposInput.d_queries = d_reference_sequences.data();
            startendposInput.d_queryOffsets = d_reference_offsets.data(),
            startendposInput.d_queryLengths = d_reference_lengths.data();
            startendposInput.maximumSubjectLength = readSequences.maximumSequenceLength;
            startendposInput.maximumQueryLength = referenceSequences.maximumSequenceLength;
            startendposInput.numAlignments = totalNumBestResults;
            startendposInput.d_segmentIdsPerElement = d_segmentIdsPerElement.data();
            startendposInput.d_referenceSequenceIdsOfMax = d_referenceSequenceIdsOfMaxScore.data();
            startendposInput.d_numReferenceSequenceIdsPerReadPrefixSum = d_numBestResultsPerReadPrefixSum.data();
    
            helpers::GpuTimer timerStartEndpos(mainStream, "batch "  + std::to_string(batchDataPtr->batchId) + " startendpos for bests kernels");
            computeAlignmentsStartAndEndpositions_localAlignment_floatOrInt_multitile<alphabetSize,float,PenaltyType::Affine,512,8,20>(
                d_bestScores_startendpos.data(),
                d_queryEndPositions_inclusive.data(),
                d_subjectEndPositions_inclusive.data(),
                d_queryStartPositions_inclusive.data(),
                d_subjectStartPositions_inclusive.data(),
                startendposInput,
                gpuSubstitutionMatrix.getSubmat32(),
                options.scoring,
                readSequences.maximumSequenceLength,
                d_temp.data(),
                d_temp.size(),
                mainStream
            );
            timerStartEndpos.stop();
            if(options.verbose){
                timerStartEndpos.print();
            }

            // check if scores are equal to the maximum scores
            // rmm::device_uvector<int> d_expectedScores(totalNumBestResults, mainStream);
            // thrust::gather(
            //     rmm::exec_policy_nosync(mainStream),
            //     d_segmentIdsPerElement.begin(), 
            //     d_segmentIdsPerElement.end(), 
            //     d_maxScores.begin(),
            //     d_expectedScores.begin()
            // );

            // bool samescores = thrust::equal(
            //     rmm::exec_policy_nosync(mainStream),
            //     d_expectedScores.begin(), d_expectedScores.end(),
            //     d_bestScores_startendpos.begin()
            // );
            // std::cout << "startendpos only for best aligments produces same score: " << samescores << "\n";
            
        }else if(GpuAligner::alignmentType == AlignmentType::SemiglobalAlignment){

            //compute the start and end positions for the best scoring alignments per read

            rmm::device_uvector<int> d_bestScores_startendpos(totalNumBestResults, mainStream);
            d_temp.resize(size_t(1) << 31, mainStream);

            StartEndPosOfBestInputDataForSubstmat startendposInput;
            startendposInput.numSubjects = numReadSequences;
            startendposInput.numQueries = numReferenceSequences;
            startendposInput.d_subjects = d_read_sequences.data();
            startendposInput.d_subjectOffsets = d_read_offsets.data();
            startendposInput.d_subjectLengths = d_read_lengths.data();
            startendposInput.d_queries = d_reference_sequences.data();
            startendposInput.d_queryOffsets = d_reference_offsets.data(),
            startendposInput.d_queryLengths = d_reference_lengths.data();
            startendposInput.maximumSubjectLength = readSequences.maximumSequenceLength;
            startendposInput.maximumQueryLength = referenceSequences.maximumSequenceLength;
            startendposInput.numAlignments = totalNumBestResults;
            startendposInput.d_segmentIdsPerElement = d_segmentIdsPerElement.data();
            startendposInput.d_referenceSequenceIdsOfMax = d_referenceSequenceIdsOfMaxScore.data();
            startendposInput.d_numReferenceSequenceIdsPerReadPrefixSum = d_numBestResultsPerReadPrefixSum.data();
    
            helpers::GpuTimer timerStartEndpos(mainStream, "batch "  + std::to_string(batchDataPtr->batchId) + " startendpos for bests kernels");
            computeAlignmentsStartAndEndpositions_semiglobalAlignment_floatOrInt_multitile<alphabetSize,float,PenaltyType::Affine,512,8,20>(
                d_bestScores_startendpos.data(),
                d_queryEndPositions_inclusive.data(),
                d_subjectEndPositions_inclusive.data(),
                d_queryStartPositions_inclusive.data(),
                d_subjectStartPositions_inclusive.data(),
                startendposInput,
                gpuSubstitutionMatrix.getSubmat32(),
                options.scoring,
                readSequences.maximumSequenceLength,
                d_temp.data(),
                d_temp.size(),
                mainStream
            );
            timerStartEndpos.stop();
            if(options.verbose){
                timerStartEndpos.print();
            }

            // check if scores are equal to the maximum scores
            // rmm::device_uvector<int> d_expectedScores(totalNumBestResults, mainStream);
            // thrust::gather(
            //     rmm::exec_policy_nosync(mainStream),
            //     d_segmentIdsPerElement.begin(), 
            //     d_segmentIdsPerElement.end(), 
            //     d_maxScores.begin(),
            //     d_expectedScores.begin()
            // );

            // bool samescores = thrust::equal(
            //     rmm::exec_policy_nosync(mainStream),
            //     d_expectedScores.begin(), d_expectedScores.end(),
            //     d_bestScores_startendpos.begin()
            // );
            // std::cout << "startendpos only for best aligments produces same score: " << samescores << "\n";
        }

        // for(int i = 0; i < numReadSequences; i++){
        //     const int numMax = d_numBestResultsPerRead.element(i, mainStream);
        //     if(numMax > 50){
        //         std::cout << "batch " << batchDataPtr->batchId << ", read " << i << ", numMax " << numMax << "\n";
        //         std::exit(0);
        //     }
        // }


        batchDataPtr->h_scores.resize(numReadSequences);
        batchDataPtr->h_readIndexPerReferenceSequence.resize(totalNumBestResults);
        batchDataPtr->h_referenceSequenceIdsOfMax.resize(totalNumBestResults);
        batchDataPtr->h_queryStartPositions_inclusive.resize(totalNumBestResults);
        batchDataPtr->h_subjectStartPositions_inclusive.resize(totalNumBestResults);
        batchDataPtr->h_queryEndPositions_inclusive.resize(totalNumBestResults);
        batchDataPtr->h_subjectEndPositions_inclusive.resize(totalNumBestResults);
        batchDataPtr->h_permutationlist.resize(numReadSequences);
        batchDataPtr->h_numReferenceSequenceIdsPerRead.resize(numReadSequences);
        batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum.resize(numReadSequences+1);


        CUDACHECK(cudaEventRecord(batchDataPtr->d2h_Start, mainStream));
        {
            // repurpose d_read_lengths since it is no longer needed
            int* const d_permutationlist = d_read_lengths.data();
            thrust::scatter(
                rmm::exec_policy_nosync(mainStream), 
                thrust::make_counting_iterator(0), 
                thrust::make_counting_iterator(numReadSequences), 
                d_originalInputPositions.begin(), 
                d_permutationlist
            );
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_permutationlist.data(), d_permutationlist, sizeof(int) * numReadSequences, cudaMemcpyDeviceToHost, mainStream));
            
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_scores.data(), d_maxScores.data(), sizeof(int) * numReadSequences, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_queryStartPositions_inclusive.data(), d_queryStartPositions_inclusive.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_subjectStartPositions_inclusive.data(), d_subjectStartPositions_inclusive.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_queryEndPositions_inclusive.data(), d_queryEndPositions_inclusive.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_subjectEndPositions_inclusive.data(), d_subjectEndPositions_inclusive.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));

            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_readIndexPerReferenceSequence.data(), d_segmentIdsPerElement.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_referenceSequenceIdsOfMax.data(), d_referenceSequenceIdsOfMaxScore.data(), sizeof(int) * totalNumBestResults, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_numReferenceSequenceIdsPerRead.data(), d_numBestResultsPerRead.data(), sizeof(int) * numReadSequences, cudaMemcpyDeviceToHost, mainStream));
            CUDACHECK(cudaMemcpyAsync(batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum.data(), d_numBestResultsPerReadPrefixSum.data(), sizeof(int) * numReadSequences, cudaMemcpyDeviceToHost, mainStream));
            
            CUDACHECK(cudaEventRecord(batchDataPtr->d2h_Stop, mainStream));

            // batchDataPtr->computedCells = 0;
            // for(size_t i = 0; i < numReadSequences; i++){
            //     batchDataPtr->computedCells += readSequences.h_lengths[i] * readSequences.h_lengths[i];
            // }
        }
    
    }

};


#endif