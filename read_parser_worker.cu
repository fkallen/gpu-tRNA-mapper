#include "execution_pipeline.cuh"
#include "common.cuh"
#include "parsing.cuh"

#include "hpc_helpers/timers.cuh"

#include <future>
#include <iostream>

#include <nvtx3/nvToolsExt.h>

struct ReadParserWorker{
    const Options* optionsPtr;
    BatchDataQueue* batchDataQueueIn;
    BatchDataQueue* batchDataQueueOut;

    ReadParserWorker(const Options* optionsPtr_, BatchDataQueue* batchDataQueueIn_, BatchDataQueue* batchDataQueueOut_)
        : optionsPtr(optionsPtr_), batchDataQueueIn(batchDataQueueIn_), batchDataQueueOut(batchDataQueueOut_)
    {

    }

    void run(){
        nvtx3::scoped_range sr("ReadParserWorker::run");
        helpers::CpuTimer cputimer("ReadParserWorker::run");
        const auto& options = *optionsPtr;
        constexpr int withHeaders = true;

        const bool hasQualityScores = [&](){
            kseqpp::KseqPP reader(options.readFileName);
            if(reader.next() > 0){
                return reader.getCurrentQuality().size() > 0;
            }
            return false;
        }();

    
        const int maxbatchsize = options.batchsize;
        kseqpp::KseqPP reader(options.readFileName);

        std::int64_t totalParsedSequences = 0;
        std::int64_t totalSubmittedBatches = 0;
    
        BatchData* batchDataPtr = batchDataQueueIn->pop();
        int parsed = 0;
        do{
            batchDataPtr->batchId = totalSubmittedBatches;
            helpers::CpuTimer batchtimer("batch " + std::to_string(batchDataPtr->batchId) + " parsing");
            batchDataPtr->readSequences.clear();
            batchDataPtr->readSequences.hasQualityScores = hasQualityScores;
            parsed = parse_n_sequences(reader, batchDataPtr->readSequences, maxbatchsize, withHeaders, hasQualityScores);
            if(options.verbose){
                batchtimer.print();
            }
            if(parsed > 0){
                // std::cout << "parsed " << parsed << "\n";
                totalParsedSequences += parsed;
                // nvtx3::mark("batchDataQueueOut->push");
                batchDataQueueOut->push(batchDataPtr);
                totalSubmittedBatches++;
                batchDataPtr = batchDataQueueIn->pop();
            }
        }while((parsed == maxbatchsize) && (options.numBatches != totalSubmittedBatches));

        batchDataQueueOut->push(nullptr); //notify consumer that parsing is complete
    
        cputimer.stop();
    
        // std::cout << "parsedSequences " << totalParsedSequences << "\n";
        // std::cout << "submittedBatches " << totalSubmittedBatches << "\n";

        if(options.verbose){
            cputimer.print();
        }
    }
};

std::future<void> launchReadParser(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    int deviceId
){
    return std::async(std::launch::async,
        [=](){
            try{
                CUDACHECK(cudaSetDevice(deviceId));
                ReadParserWorker worker(options, inputQueue, outputQueue);                   
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in read parser\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}