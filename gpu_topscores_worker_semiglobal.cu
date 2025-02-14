#include "gpu_topscores_worker.cuh"
#include "execution_pipeline.cuh"
#include "parasail_helpers.hpp"

#include "common.cuh"
#include "hpc_helpers/timers.cuh"

#include <future>
#include <iostream>

#include <nvtx3/nvToolsExt.h>

#include "cuda_errorcheck.cuh"
#include <gpu_api/semiglobal_alignment_api.cuh>
#include <gpu_api/length_switcher_api.cuh>


std::future<void> launchSemiglobalAlignmentGPUTopscoresWorker(
    const Options* options,
    BatchDataQueue* inputQueue,
    BatchDataQueue* outputQueue,
    const std::vector<std::vector<int>>* substitutionMatrix2D,
    parasail_matrix_t* parasailScoringMatrix,
    int deviceId
){
    constexpr auto penaltyType = PenaltyType::Affine;

    return std::async(std::launch::async,
        [=](){
            constexpr int blocksize = 512;
            using ListOfConfigsShortQuery = ListOfTypes<
                // KernelConfig<blocksize, 4, 4>,
                // KernelConfig<blocksize, 4, 8>,
                // KernelConfig<blocksize, 4, 12>,
                // KernelConfig<blocksize, 4, 16>,
                // KernelConfig<blocksize, 4, 20>,
                // KernelConfig<blocksize, 4, 24>,
                KernelConfig<blocksize, 8, 16>,
                KernelConfig<blocksize, 8, 20>,
                // KernelConfig<blocksize, 8, 24>,
                // KernelConfig<blocksize, 8, 28>,
                // KernelConfig<blocksize, 8, 32>
                // KernelConfig<blocksize, 8, 16>,
                // KernelConfig<blocksize, 8, 20>
            >;
            constexpr int blocksize2 = 256;
            using ListOfConfigsLongQuery = ListOfTypes<
                KernelConfig<blocksize2, 16, 8>
            >;

            using NoConfigs = ListOfTypes<>;
            
            try{
                CUDACHECK(cudaSetDevice(deviceId));

                using GpuAlignerSwitch = GpuAlignerWithLengthSwitch<
                    SemiglobalAlignment_32bit,
                    alphabetSize,
                    ScoreType32,
                    penaltyType,
                    ListOfConfigsShortQuery,
                    ListOfConfigsLongQuery
                >;
                GpuTopScoresAlignerWorker<GpuAlignerSwitch> worker(
                    options, 
                    *substitutionMatrix2D,
                    parasailScoringMatrix,
                    inputQueue, 
                    outputQueue
                );
                worker.run();


            }catch (const rmm::bad_alloc& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in gpu topscores aligner semiglobal\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}