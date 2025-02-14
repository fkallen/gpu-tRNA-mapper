#include "execution_pipeline.cuh"

#include "common.cuh"
#include "hpc_helpers/timers.cuh"

#include "sam_helpers.hpp"

#include <future>
#include <iostream>

#include <nvtx3/nvToolsExt.h>

struct CpuTracebackAlignerWorker{

    const Options* optionsPtr;
    BatchDataQueue* batchDataQueueIn;
    BatchDataQueue* batchDataQueueOut;

    std::vector<std::vector<int>> substitutionMatrix2D;
    parasail_matrix_t* parasail_scoring_matrix;

    CpuTracebackAlignerWorker(
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
        parasail_scoring_matrix(parasail_scoring_matrix_)
    {
    }

    void run(){
        nvtx3::scoped_range sr("CpuTracebackAlignerWorker::run");
        helpers::CpuTimer cputimer("CpuTracebackAlignerWorker::run");

        const auto& options = *optionsPtr;
       
        BatchData* batchDataPtr = batchDataQueueIn->pop();

        while(batchDataPtr != nullptr){

            applyResultListSize(batchDataPtr);

            if(options.alignmentType == AlignmentType::LocalAlignment){
                helpers::CpuTimer timer1("batch " + std::to_string(batchDataPtr->batchId) + " computeTracebacksOfBestScoringAlignments_SW");
                computeTracebacksOfBestScoringAlignments_SW(
                    batchDataPtr
                );
                if(options.verbose){
                    timer1.print();
                }
            }else if(options.alignmentType == AlignmentType::SemiglobalAlignment){
                helpers::CpuTimer timer1("batch " + std::to_string(batchDataPtr->batchId) + " computeTracebacksOfBestScoringAlignments_SG");
                computeTracebacksOfBestScoringAlignments_SG(
                    batchDataPtr
                );
                if(options.verbose){
                    timer1.print();
                }
            }

            

            if(options.alignmentType == AlignmentType::LocalAlignment){
                helpers::CpuTimer timer2("batch " + std::to_string(batchDataPtr->batchId) + " convertResultsToSAMstrings_SW");
                convertResultsToSAMstrings_SW(
                    batchDataPtr
                );
                if(options.verbose){
                    timer2.print();
                }
            }else if(options.alignmentType == AlignmentType::SemiglobalAlignment){
                helpers::CpuTimer timer2("batch " + std::to_string(batchDataPtr->batchId) + " convertResultsToSAMstrings_SG");
                convertResultsToSAMstrings_SG(
                    batchDataPtr
                );
                if(options.verbose){
                    timer2.print();
                }
            }

            batchDataQueueOut->push(batchDataPtr);
            batchDataPtr = batchDataQueueIn->pop();
        }

        //notify writer
        batchDataQueueOut->push(nullptr);

        if(options.verbose){
            cputimer.print();
        }
    }

    void applyResultListSize(
        BatchData* batchDataPtr
    ){
        const SequenceCollection& readSequences = batchDataPtr->readSequences;
        const int numReadSequences = readSequences.h_lengths.size();

        for(int a = 0; a < numReadSequences; a++){
            const int perm = batchDataPtr->h_permutationlist[a];
            const int x = batchDataPtr->h_numReferenceSequenceIdsPerRead[perm];
            batchDataPtr->h_numReferenceSequenceIdsPerRead[perm] = std::min(optionsPtr->resultListSize, x);
        }
    }

    void computeTracebacksOfBestScoringAlignments_SW(
        BatchData* batchDataPtr
    ){
        nvtx3::scoped_range sr("computeTracebacksOfBestScoringAlignments_SW");

        const auto& options = *optionsPtr;
    
        const SequenceCollection& readSequences = batchDataPtr->readSequences;
        const SequenceCollection& referenceSequences = *batchDataPtr->referenceSequences;

        const int numReadSequences = readSequences.h_lengths.size();
        const int numReferenceSequences = referenceSequences.h_lengths.size();
    
        const int totalNumBestResults = batchDataPtr->h_referenceSequenceIdsOfMax.size();

        // batchDataPtr->parasailResultData.clear();
        // batchDataPtr->parasailResultData.reserve(totalNumBestResults);
        // for(int i = 0; i < totalNumBestResults; i++){
        //     batchDataPtr->parasailResultData.emplace_back();
        // }

        batchDataPtr->parasailResultData.resize(totalNumBestResults);


        {
            // helpers::CpuTimer tracebackTimer("cpu traceback");
            // nvtx3::scoped_range range("cpu traceback");
            #pragma omp parallel for schedule(dynamic)
            for(int a = 0; a < numReadSequences; a++){
                const int perm = batchDataPtr->h_permutationlist[a];
                const size_t readOffset = readSequences.h_offsets[a];
                const char* readSequence = readSequences.h_sequences.data() + readOffset;
                const int readLength = readSequences.h_lengths[a];

                const int numResultsForRead = batchDataPtr->h_numReferenceSequenceIdsPerRead[perm];
                const int resultOffset = batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum[perm];
                auto* resultsForRead = batchDataPtr->parasailResultData.data() + resultOffset;
                const int score = batchDataPtr->h_scores[perm];
                // const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdOfMax[perm];

                
                for(int resultId = 0; resultId < numResultsForRead; resultId++){
                    const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[resultOffset + resultId];
                    const size_t referenceOffset = referenceSequences.h_offsets[referenceSequenceId];
                    const char* referenceSequence = referenceSequences.h_sequences.data() + referenceOffset;
                    const int referenceLength = referenceSequences.h_lengths[referenceSequenceId];

                    const int refBeginPos = batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId];
                    const int refEndPos = batchDataPtr->h_queryEndPositions_inclusive[resultOffset + resultId] + 1;
                    const int readBeginPos = batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId];
                    const int readEndPos = batchDataPtr->h_subjectEndPositions_inclusive[resultOffset + resultId] + 1;

                    auto NW_result = ParasailResult(
                        parasail_nw_trace_diag_32(
                            readSequence + readBeginPos, (readEndPos - readBeginPos),
                            referenceSequence + refBeginPos, (refEndPos - refBeginPos),
                            std::abs(options.scoring.gapopenscore), std::abs(options.scoring.gapextendscore),
                            parasail_scoring_matrix
                        ),
                        parasail_result_free
                    );
        
                    if(parasail_result_get_score(NW_result.get()) != score){
                        std::cout << "startendpos score mismatch to parasail: score " << score << ", parasail " << parasail_result_get_score(NW_result.get()) << "\n";
                        std::cout << "read:\n";
                        for(int i = 0; i < readLength; i++){
                            std::cout << readSequence[i];
                        }
                        std::cout << "\n";
                        std::cout << "reference:\n";
                        for(int i = 0; i < referenceLength; i++){
                            std::cout << referenceSequence[i];
                        }
                        std::cout << "\n";
                        // std::cout << "refBeginPos " << refBeginPos << ", refEndPos " << refEndPos << "\n";
                        // std::cout << "readBeginPos " << readBeginPos << ", readEndPos " << readEndPos << "\n";
                        // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
                        //     parasail_result_get_cigar_extra(
                        //         // SW_result.get(),
                        //         NW_result.get(),
                        //         readSequence, readLength,
                        //         referenceSequence, referenceLength,
                        //         parasail_scoring_matrix
                        // ,
                        //     true,
                        //     "TU"
                        //     ),
                        //     parasail_cigar_free
                        // );
                        // auto parasailCigarString = std::unique_ptr<char, void (*)(void*)>(
                        //     parasail_cigar_decode(parasailCigar.get()),
                        //     free
                        // );
                        // std::cout << "best start end pos of parasail:\n";
                        // std::cout << "refBeginPos " << parasailCigar->beg_ref << ", refEndPos " << SW_result->end_ref+1 << "\n";
                        // std::cout << "readBeginPos " << parasailCigar->beg_query << ", readEndPos " << SW_result->end_query+1 << "\n";
                        // std::cout << "Cigar: " << parasailCigarString.get() << "\n";
                        
                        std::exit(0);
                    }
                    auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
                        parasail_result_get_cigar_extra(
                            NW_result.get(),
                            readSequence + readBeginPos, (readEndPos - readBeginPos),
                            referenceSequence + refBeginPos, (refEndPos - refBeginPos),
                            parasail_scoring_matrix,
                            true,
                            "TU"
                        ),
                        parasail_cigar_free
                    );
                    resultsForRead[resultId].result = std::move(NW_result);
                    resultsForRead[resultId].cigar = std::move(parasailCigar);
                }

            }


            // tracebackTimer.print();
        }

    }

    void computeTracebacksOfBestScoringAlignments_SG(
        BatchData* batchDataPtr
    ){
        nvtx3::scoped_range sr("computeTracebacksOfBestScoringAlignments_SG");

        const auto& options = *optionsPtr;
    
        const SequenceCollection& readSequences = batchDataPtr->readSequences;
        const SequenceCollection& referenceSequences = *batchDataPtr->referenceSequences;

        const int numReadSequences = readSequences.h_lengths.size();
        const int numReferenceSequences = referenceSequences.h_lengths.size();
    
        const int totalNumBestResults = batchDataPtr->h_referenceSequenceIdsOfMax.size();

        // batchDataPtr->parasailResultData.clear();
        // batchDataPtr->parasailResultData.reserve(totalNumBestResults);
        // for(int i = 0; i < totalNumBestResults; i++){
        //     batchDataPtr->parasailResultData.emplace_back();
        // }

        batchDataPtr->parasailResultData.resize(totalNumBestResults);
        {
            // helpers::CpuTimer tracebackTimer("cpu traceback");
            // nvtx3::scoped_range range("cpu traceback");

            #pragma omp parallel for schedule(dynamic)
            for(int a = 0; a < numReadSequences; a++){
                const int perm = batchDataPtr->h_permutationlist[a];
                const size_t readOffset = readSequences.h_offsets[a];
                const char* readSequence = readSequences.h_sequences.data() + readOffset;
                const int readLength = readSequences.h_lengths[a];

                const int numResultsForRead = batchDataPtr->h_numReferenceSequenceIdsPerRead[perm];
                const int resultOffset = batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum[perm];
                auto* resultsForRead = batchDataPtr->parasailResultData.data() + resultOffset;
                const int score = batchDataPtr->h_scores[perm];
                // const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdOfMax[perm];

                
                for(int resultId = 0; resultId < numResultsForRead; resultId++){
                    const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[resultOffset + resultId];
                    const size_t referenceOffset = referenceSequences.h_offsets[referenceSequenceId];
                    const char* referenceSequence = referenceSequences.h_sequences.data() + referenceOffset;
                    const int referenceLength = referenceSequences.h_lengths[referenceSequenceId];

                    const int refBeginPos = batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId];
                    const int refEndPos_excl = batchDataPtr->h_queryEndPositions_inclusive[resultOffset + resultId] + 1;
                    const int readBeginPos = batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId];
                    const int readEndPos_excl = batchDataPtr->h_subjectEndPositions_inclusive[resultOffset + resultId] + 1;

                    // const int refBeginPos = std::max(0, batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId]);
                    // const int refEndPos_excl = batchDataPtr->h_queryEndPositions_inclusive[resultOffset + resultId] + 1;
                    // const int readBeginPos = std::max(0, batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId]);
                    // const int readEndPos_excl = batchDataPtr->h_subjectEndPositions_inclusive[resultOffset + resultId] + 1;


                    const bool beginsAtFirstRowOrCol = (refBeginPos == 0) || (readBeginPos == 0);
                    const bool endsAtLastRowOrCol = (refEndPos_excl == referenceLength) || (readEndPos_excl == readLength);
                    if(!(beginsAtFirstRowOrCol && endsAtLastRowOrCol)){
                        std::cerr << "start endpos are impossible for semi global.\n";
                        std::cerr << "refBeginPos " << refBeginPos << ", refEndPos " << refEndPos_excl << ", readBeginPos " << readBeginPos << ", readEndPos " << readEndPos_excl << "\n";
                    }

                    //do not penalize gaps and the begin of sequences (start positions are not being computed yet, they are 0 0)
                    //at sequence end, we do not allow free gaps as we already determined the end positions

                    parasail_result_t* resultptr = nullptr;
                    if(batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId] == -1 || batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId] == -1){
                        std::cerr << "error start pos -1\n";
                        std::exit(0);

                    }else{
                        resultptr = parasail_nw_trace_diag_32(
                            readSequence + readBeginPos, (readEndPos_excl - readBeginPos),
                            referenceSequence + refBeginPos, (refEndPos_excl - refBeginPos),
                            std::abs(options.scoring.gapopenscore), std::abs(options.scoring.gapextendscore),
                            parasail_scoring_matrix
                        );
                    }
                    auto SG_result = ParasailResult(resultptr, parasail_result_free);
                                            
        
                    if(parasail_result_get_score(SG_result.get()) != score){
                        for(int i = 0; i < (readEndPos_excl - readBeginPos); i++){
                            std::cout << readSequence[readBeginPos + i];
                        }
                        std::cout << "\n";
                        for(int i = 0; i < (refEndPos_excl - refBeginPos); i++){
                            std::cout << referenceSequence[refBeginPos + i];
                        }
                        std::cout << "\n";

                        auto SG_endpos_table_result = ParasailResult(
                            parasail_sg_qb_db_table(
                                readSequence + readBeginPos, (readEndPos_excl - readBeginPos),
                                referenceSequence + refBeginPos, (refEndPos_excl - refBeginPos),
                                std::abs(options.scoring.gapopenscore), std::abs(options.scoring.gapextendscore),
                                parasail_scoring_matrix
                            ),
                            parasail_result_free
                        );

                        auto SG_result_full = ParasailResult(
                            parasail_sg_trace_striped_32(
                                readSequence, readLength,
                                referenceSequence, referenceLength,
                                std::abs(options.scoring.gapopenscore), std::abs(options.scoring.gapextendscore),
                                parasail_scoring_matrix
                            ),
                            parasail_result_free
                        );

                        auto SG_full_table_result = ParasailResult(
                            parasail_sg_table(
                                readSequence, readLength,
                                referenceSequence, referenceLength,
                                std::abs(options.scoring.gapopenscore), std::abs(options.scoring.gapextendscore),
                                parasail_scoring_matrix
                            ),
                            parasail_result_free
                        ); 

                        std::cerr << "score " << score << ", refBeginPos " << refBeginPos << ", refEndPos " << refEndPos_excl << ", readBeginPos " << readBeginPos << ", readEndPos " << readEndPos_excl << "\n";

                        std::cout << "startendpos score mismatch to parasail: score " << score 
                            << ", parasail " << parasail_result_get_score(SG_result.get()) 
                            << ", parasail full " << parasail_result_get_score(SG_result_full.get())                                     
                            << "\n";
                        std::cout << "read:\n";
                        for(int i = 0; i < readLength; i++){
                            std::cout << readSequence[i];
                        }
                        std::cout << "\n";
                        std::cout << "reference:\n";
                        for(int i = 0; i < referenceLength; i++){
                            std::cout << referenceSequence[i];
                        }
                        std::cout << "\n";

                        int* fulltable = parasail_result_get_score_table(SG_full_table_result.get()); 
                        // std::cout << " parasail full score at endpos position " << fulltable[(readEndPos_excl-1) * referenceLength + (refEndPos_excl-1)] << "\n";

                        // std::cout << "SG_full_table_result\n";
                        // for(int r = 0; r < readLength; r++){
                        //     for(int c = 0; c < referenceLength; c++){
                        //         std::cout << fulltable[r * referenceLength + c] << " ";
                        //     }
                        //     std::cout << "\n";
                        // }
                        // std::cout << "\n";

                        // int* endpostable = parasail_result_get_score_table(SG_endpos_table_result.get()); 
                        // std::cout << "SG_endpos_table_result\n";
                        // for(int r = 0; r < readEndPos_excl; r++){
                        //     for(int c = 0; c < refEndPos_excl; c++){
                        //         std::cout << endpostable[r * refEndPos_excl + c] << " ";
                        //     }
                        //     std::cout << "\n";
                        // }
                        // std::cout << "\n";
                        
                        std::exit(0);
                    }
                    auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
                        parasail_result_get_cigar_extra(
                            SG_result.get(),
                            readSequence + readBeginPos, (readEndPos_excl - readBeginPos),
                            referenceSequence + refBeginPos, (refEndPos_excl - refBeginPos),
                            parasail_scoring_matrix,
                            true,
                            "TU"
                        ),
                        parasail_cigar_free
                    );
                    resultsForRead[resultId].result = std::move(SG_result);
                    resultsForRead[resultId].cigar = std::move(parasailCigar);                    
                }

            }

          


            // tracebackTimer.print();
        }

    }

    void convertResultsToSAMstrings_SW(
        BatchData* batchDataPtr
    ){
        nvtx3::scoped_range sr("convertResultsToSAMstrings_SW");

        // const auto& options = *optionsPtr;

        const int numReads = batchDataPtr->readSequences.h_lengths.size();
    
        const int totalNumBestResults = batchDataPtr->h_referenceSequenceIdsOfMax.size();
        batchDataPtr->samresultStrings.resize(totalNumBestResults);
        for(auto& s : batchDataPtr->samresultStrings){
            s.clear();
        }

        #pragma omp parallel for schedule(dynamic)
        for(int a = 0; a < numReads; a++){
            const int perm = batchDataPtr->h_permutationlist[a];
            // const int score = batchDataPtr->h_scores[perm];
            // const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[perm];

            const auto offset1 = batchDataPtr->readSequences.h_offsets[a];
            const char* readSequence = batchDataPtr->readSequences.h_sequences.data() + offset1;
            const int readLength = batchDataPtr->readSequences.h_lengths[a];
            auto readSequenceView = std::string_view(readSequence, readLength);
            const char* readQuality = batchDataPtr->readSequences.qualities.data() + offset1;
            const int qualityLength = batchDataPtr->readSequences.hasQualityScores ? readLength : 0;
            auto readQualityView = std::string_view(readQuality, qualityLength);

            const int numResultsForRead = batchDataPtr->h_numReferenceSequenceIdsPerRead[perm];
            const int resultOffset = batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum[perm];
            const auto* resultsForRead = batchDataPtr->parasailResultData.data() + resultOffset;
            std::string* samstringsForRead = batchDataPtr->samresultStrings.data() + resultOffset;
            // const int score = batchDataPtr->h_scores[perm];
            // const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdOfMax[perm];

            
            for(int resultId = 0; resultId < numResultsForRead; resultId++){
                const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[resultOffset + resultId];
                const size_t referenceOffset = batchDataPtr->referenceSequences->h_offsets[referenceSequenceId];
                const char* referenceSequence = batchDataPtr->referenceSequences->h_sequences.data() + referenceOffset;
                const int referenceLength = batchDataPtr->referenceSequences->h_lengths[referenceSequenceId];

                const int readBeginPos = batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId];
                const int readEndPos_excl = batchDataPtr->h_subjectEndPositions_inclusive[resultOffset + resultId] + 1;
                const int refBeginPos = batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId];
                const int refEndPos_excl = batchDataPtr->h_queryEndPositions_inclusive[resultOffset + resultId] + 1;                        

                appendAlignmentResultInSAMformat_NW_to_SW(
                    samstringsForRead[resultId],
                    batchDataPtr->readSequences.sequenceNames[a], 
                    batchDataPtr->referenceSequences->sequenceNames[referenceSequenceId],
                    readSequenceView,
                    referenceSequence,
                    readQualityView,
                    readBeginPos,
                    readEndPos_excl,
                    refBeginPos,
                    refEndPos_excl,
                    resultsForRead[resultId],
                    parasail_scoring_matrix
                );
            }            
        }
    }


    void convertResultsToSAMstrings_SG(
        BatchData* batchDataPtr
    ){
        nvtx3::scoped_range sr("convertResultsToSAMstrings_SG");

        // const auto& options = *optionsPtr;

        const int numReads = batchDataPtr->readSequences.h_lengths.size();
    
        const int totalNumBestResults = batchDataPtr->h_referenceSequenceIdsOfMax.size();
        batchDataPtr->samresultStrings.resize(totalNumBestResults);
        for(auto& s : batchDataPtr->samresultStrings){
            s.clear();
        }

        #pragma omp parallel for schedule(dynamic)
        for(int a = 0; a < numReads; a++){
            const int perm = batchDataPtr->h_permutationlist[a];
            const int score = batchDataPtr->h_scores[perm];
            // const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[perm];

            const auto offset1 = batchDataPtr->readSequences.h_offsets[a];
            const char* readSequence = batchDataPtr->readSequences.h_sequences.data() + offset1;
            const int readLength = batchDataPtr->readSequences.h_lengths[a];
            auto readSequenceView = std::string_view(readSequence, readLength);
            const char* readQuality = batchDataPtr->readSequences.qualities.data() + offset1;
            const int qualityLength = batchDataPtr->readSequences.hasQualityScores ? readLength : 0;
            auto readQualityView = std::string_view(readQuality, qualityLength);

            const int numResultsForRead = batchDataPtr->h_numReferenceSequenceIdsPerRead[perm];
            const int resultOffset = batchDataPtr->h_numReferenceSequenceIdsPerReadPrefixSum[perm];
            const auto* resultsForRead = batchDataPtr->parasailResultData.data() + resultOffset;
            std::string* samstringsForRead = batchDataPtr->samresultStrings.data() + resultOffset;

            
            for(int resultId = 0; resultId < numResultsForRead; resultId++){
                const int referenceSequenceId = batchDataPtr->h_referenceSequenceIdsOfMax[resultOffset + resultId];
                const size_t referenceOffset = batchDataPtr->referenceSequences->h_offsets[referenceSequenceId];
                const char* referenceSequence = batchDataPtr->referenceSequences->h_sequences.data() + referenceOffset;
                const int referenceLength = batchDataPtr->referenceSequences->h_lengths[referenceSequenceId];
                    
                const int readBeginPos = batchDataPtr->h_subjectStartPositions_inclusive[resultOffset + resultId];
                const int readEndPos_excl = batchDataPtr->h_subjectEndPositions_inclusive[resultOffset + resultId] + 1;
                const int refBeginPos = batchDataPtr->h_queryStartPositions_inclusive[resultOffset + resultId];
                const int refEndPos_excl = batchDataPtr->h_queryEndPositions_inclusive[resultOffset + resultId] + 1;

                appendAlignmentResultInSAMformat_NW_to_SG(
                    samstringsForRead[resultId],
                    batchDataPtr->readSequences.sequenceNames[a], 
                    batchDataPtr->referenceSequences->sequenceNames[referenceSequenceId],
                    readSequenceView,
                    referenceSequence,
                    readQualityView,
                    readBeginPos,
                    readEndPos_excl,
                    refBeginPos,
                    refEndPos_excl,
                    score,
                    resultsForRead[resultId],
                    parasail_scoring_matrix
                );
            }            
        }
    }
};


std::future<void> launchCPUTracebackWorker(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    const std::vector<std::vector<int>>* substitutionMatrix2D,
    parasail_matrix_t* parasailScoringMatrix
){
    return std::async(std::launch::async,
        [=](){
            try{
                CpuTracebackAlignerWorker worker(
                    options, 
                    *substitutionMatrix2D,
                    parasailScoringMatrix,
                    inputQueue,
                    outputQueue
                );
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in cpu traceback worker\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}