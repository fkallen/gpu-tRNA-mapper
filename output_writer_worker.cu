#include "execution_pipeline.cuh"

#include "common.cuh"
#include "sam_helpers.hpp"
#include "hpc_helpers/timers.cuh"

#include <future>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string_view>

#include <nvtx3/nvtx3.hpp>


struct OutputWriterWorker{
    int processedNumReads = 0;
    const Options* optionsPtr;
    BatchDataQueue* batchDataQueueIn;
    BatchDataQueue* batchDataQueueOut;

    OutputWriterWorker(const Options* optionsPtr_, BatchDataQueue* batchDataQueueIn_, BatchDataQueue* batchDataQueueOut_)
        : optionsPtr(optionsPtr_), batchDataQueueIn(batchDataQueueIn_), batchDataQueueOut(batchDataQueueOut_)
    {

    }

    void run(){
        nvtx3::scoped_range sr("OutputWriterWorker::run");
        helpers::CpuTimer cputimer("OutputWriterWorker::run");
        const auto& options = *optionsPtr;

        std::ofstream outputfilestream;
        bool outputToFile = false;
        if(options.outputFileName != ""){
            outputfilestream.open(options.outputFileName);
            if(!(outputfilestream)){
                throw std::runtime_error("could not open output file");
            }
            outputToFile = true;
        }

        std::ostream* osPtr;
        if(outputToFile){
            osPtr = &outputfilestream;
        }else{
            osPtr = &std::cout;
        }

        BatchData* batchDataPtr = batchDataQueueIn->pop();

        if(batchDataPtr != nullptr){
            writeSAMheader(*osPtr, batchDataPtr->referenceSequences);
        }


        while(batchDataPtr != nullptr){

            // constructAndOutputSAMresults(batchDataPtr, *osPtr);

            helpers::CpuTimer timer2("batch " + std::to_string(batchDataPtr->batchId) + " outputPreconstructedSAMresults");
            outputPreconstructedSAMresults(batchDataPtr, *osPtr);
            if(options.verbose){
                timer2.print();
            }

            batchDataQueueOut->push(batchDataPtr);
            batchDataPtr = batchDataQueueIn->pop();
        }
    
        cputimer.stop();
        if(options.verbose){
            cputimer.print();
        }
    }



    void outputPreconstructedSAMresults(
        BatchData* batchDataPtr,
        std::ostream& outputstream
    ){
        nvtx3::scoped_range range("outputPreconstructedSAMresults");

        // const auto& options = *optionsPtr;
        const int numReads = batchDataPtr->h_permutationlist.size();

        std::stringstream sstream;
        auto reset_stringstream = [&](){
            sstream.str("");
            sstream.clear();
        };
            
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
            const std::string* samstringsForRead = batchDataPtr->samresultStrings.data() + resultOffset;
            

            if(numResultsForRead == 0){
                writeUnmappedAlignmentResultInSAMformat(
                    sstream,
                    batchDataPtr->readSequences.sequenceNames[a], 
                    readSequenceView,
                    readQualityView
                );
            }

            
            for(int resultId = 0; resultId < numResultsForRead; resultId++){                 
                sstream << samstringsForRead[resultId];
            }

            if(a % 8192 == 0){
                if(sstream.rdbuf()->in_avail() > 0){
                    outputstream << sstream.rdbuf();
                    reset_stringstream();
                }
            }
            
        }
        processedNumReads += numReads;

        if(sstream.rdbuf()->in_avail() > 0){
            outputstream << sstream.rdbuf();
            reset_stringstream();
        }

    }
};


std::future<void> launchOutputWriter(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    int deviceId
){
    return std::async(std::launch::async,
        [=](){
            try{
                CUDACHECK(cudaSetDevice(deviceId));
                OutputWriterWorker worker(options, inputQueue, outputQueue);
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in output writer\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}