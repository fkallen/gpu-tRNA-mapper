#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include "sam_helpers.hpp"
#include "parasail_helpers.hpp"
#include "common.cuh"
#include "parsing.cuh"
#include "execution_pipeline.cuh"
#include "smallkernels.cuh"
#include "alphabet.cuh"


std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}

void findBestScore(Options options){
    constexpr bool withHeaders = true;
    // constexpr bool withoutHeaders = false;
    // constexpr bool withQualityScores = true;
    constexpr bool withoutQualityScores = false;

    const size_t maxbatchsize = options.batchsize;

    std::vector<std::vector<int>> substitutionMatrix2D(alphabetSize, std::vector<int>(alphabetSize, options.scoring.mismatchscore));
    for(int i = 0; i < alphabetSize; i++){
        substitutionMatrix2D[i][i] = options.scoring.matchscore;
    }

    //N matches all
    // for(int i = 0; i < alphabetSize; i++){
    //     substitutionMatrix2D[alphabetSize-1][i] = options.scoring.matchscore;
    //     substitutionMatrix2D[i][alphabetSize-1] = options.scoring.matchscore;
    // }

    auto parasailScoringMatrix = [&](){
        auto matrix = ParasailMatrix(
            parasail_matrix_create("ACGTU", options.scoring.matchscore, options.scoring.mismatchscore),
            parasail_matrix_free
        );

        //give U same scores as T
        for(int c = 0; c < 5; c++){
            int t_value = matrix->user_matrix[3*matrix->size + c];
            parasail_matrix_set_value(matrix.get(), 4, c, t_value);
        }

        for(int r = 0; r < 5; r++){
            int t_value = matrix->user_matrix[r*matrix->size + 3];
            parasail_matrix_set_value(matrix.get(), r, 4, t_value);
        }

        return matrix;
    }();
    


    SequenceCollection referenceSequences = parseAllSequences(options.referenceFileName, withHeaders, withoutQualityScores);

    BatchDataQueue inputQueueForParser;
    BatchDataQueue inputQueueForAlignmentScores;
    BatchDataQueue inputQueueForAlignmentTracebacks;
    BatchDataQueue inputQueueForOutputWriter;

    std::vector<std::unique_ptr<BatchData>> batchDataVector;

    for(int i = 0; i < options.queue_depth; i++){
        auto data = std::make_unique<BatchData>();
        data->referenceSequences = &referenceSequences;

        data->readSequences.h_sequences.reserve(1 << 25);
        data->readSequences.h_lengths.reserve(maxbatchsize);
        data->readSequences.h_offsets.reserve(maxbatchsize);
        data->readSequences.headers.reserve(maxbatchsize);
        data->readSequences.sequenceNames.reserve(maxbatchsize);

        batchDataVector.push_back(std::move(data));
        inputQueueForParser.push(batchDataVector.back().get());
    }

    auto parserFuture = launchReadParser(&options, &inputQueueForParser, &inputQueueForAlignmentScores, 0);

    std::future<void> alignmentScoresFuture = [&](){
        if(options.alignmentType == AlignmentType::LocalAlignment){
            return launchLocalAlignmentGPUTopscoresWorker(
                &options,
                &inputQueueForAlignmentScores,
                &inputQueueForAlignmentTracebacks,
                &substitutionMatrix2D,
                parasailScoringMatrix.get(),
                0
            );
        }else if(options.alignmentType == AlignmentType::SemiglobalAlignment){
            return launchSemiglobalAlignmentGPUTopscoresWorker(
                &options,
                &inputQueueForAlignmentScores,
                &inputQueueForAlignmentTracebacks,
                &substitutionMatrix2D,
                parasailScoringMatrix.get(),
                0
            );
        }else{
            return std::future<void>{};
        }
    }();

    auto alignmentTracebackFuture = launchCPUTracebackWorker(
        &options,
        &inputQueueForAlignmentTracebacks,
        &inputQueueForOutputWriter,
        &substitutionMatrix2D,
        parasailScoringMatrix.get()
    );

    auto outputwriterFuture = launchOutputWriter(&options, &inputQueueForOutputWriter, &inputQueueForParser, 0);

    parserFuture.wait();
    alignmentScoresFuture.wait();
    alignmentTracebackFuture.wait();
    outputwriterFuture.wait();
}







int main(int argc, char** argv){

    Options options;

    bool hasScoringOptions = false;
    for(int x = 1; x < argc; x++){
        std::string argstring = argv[x];
        if(argstring == "--readFileName"){
            options.readFileName = argv[x+1];
            x++;
        }
        if(argstring == "--referenceFileName"){
            options.referenceFileName = argv[x+1];
            x++;
        }
        if(argstring == "--outputFileName"){
            options.outputFileName = argv[x+1];
            x++;
        }        
        if(argstring == "--batchsize"){
            options.batchsize = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--queue_depth"){
            options.queue_depth = std::atoi(argv[x+1]);
            x++;
        }
        // if(argstring == "--use16x2"){
        //     options.use16x2 = true;
        // }
        if(argstring == "--minAlignmentScore"){
            options.minAlignmentScore = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--numBatches"){
            options.numBatches = std::atoi(argv[x+1]);
            x++;
        }        
        if(argstring == "--scoring"){
            auto tokens = split(argv[x+1], ',');
            if(tokens.size() != 4){
                std::cout << "Usage: --scoring matchscore,mismatchscore,gapopenscore,gapextendscore\n";
                return 0;
            }
            options.scoring.matchscore = std::stoi(tokens[0]);
            options.scoring.mismatchscore = std::stoi(tokens[1]);
            options.scoring.gapopenscore = std::stoi(tokens[2]);
            options.scoring.gapextendscore = std::stoi(tokens[3]);
            //unused
            // options.scoring.gapscore = options.scoring.gapopenscore + options.scoring.gapextendscore;
            x += 1;
            hasScoringOptions = true;
        }
        if(argstring == "--localAlignment"){
            options.alignmentType = AlignmentType::LocalAlignment;
        }
        if(argstring == "--semiglobalAlignment"){
            options.alignmentType = AlignmentType::SemiglobalAlignment;
        }
        if(argstring == "--verbose"){
            options.verbose = true;
        }
        if(argstring == "--resultListSize"){
            options.resultListSize = std::atoi(argv[x+1]);
            x++;
        }
        
    }

    if(!hasScoringOptions){
        options.scoring.matchscore = 2;
        options.scoring.mismatchscore = -1;
        options.scoring.gapscore = -10;
        options.scoring.gapopenscore = -10;
        options.scoring.gapextendscore = -1;
    }

    if(options.verbose){
        std::cout << "readFileName = " << options.readFileName << "\n";
        std::cout << "referenceFileName = " << options.referenceFileName << "\n";
        std::cout << "outputFileName = " << options.outputFileName << "\n";
        std::cout << "alignmentType = " << to_string(options.alignmentType) << "\n";
        std::cout << "matchscore="  << options.scoring.matchscore << "\n";
        std::cout << "mismatchscore="  << options.scoring.mismatchscore << "\n";
        // std::cout << "gapscore="  << options.scoring.gapscore << "\n";
        std::cout << "gapopenscore="  << options.scoring.gapopenscore << "\n";
        std::cout << "gapextendscore="  << options.scoring.gapextendscore << "\n";
        std::cout << "verbose = " << options.verbose << "\n";
        // std::cout << "use16x2 = " << options.use16x2 << "\n";
        std::cout << "batchsize = " << options.batchsize << "\n";
        std::cout << "resultListSize = " << options.resultListSize << "\n";
        std::cout << "minAlignmentScore = " << options.minAlignmentScore << "\n";
    }


    if(options.readFileName == "" || options.referenceFileName == ""){
        std::cout << "Invalid input files\n";
        return 0;
    }
    if(options.outputFileName == ""){
        std::cout << "Invalid output file\n";
        return 0;
    }
    if(options.resultListSize <= 0){
        std::cout << "resultListSize must be > 0\n";
        return 0;
    }

    CUDACHECK(cudaSetDevice(0));

    // set up memory pools
    std::vector<std::unique_ptr<rmm::mr::cuda_async_memory_resource>> rmmCudaAsyncResources;
    auto resource = std::make_unique<rmm::mr::cuda_async_memory_resource>(0);
    rmm::mr::set_per_device_resource(rmm::cuda_device_id(0), resource.get());
    rmmCudaAsyncResources.push_back(std::move(resource));

    findBestScore(options);

    return 0;
}

