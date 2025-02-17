# gpu-tRNA-mapper
gpu-tRNA-mapper: GPU-accelerated read mapper to map against short tRNA sequences.


For a set of input reads and reference sequences, gpu-tRNA-mapper aligns each read to all reference sequences.
By default, an optimal local alignment is used. 
For each read, the highest scoring alignment is written to output file in SAM format. If multiple references share the same highest score all alignments are written to output file.

The computation of the all-to-all alignment scores is GPU-accelerated. After determining the highest alignment score for each read, the parasail alignment library is used on the CPU to recompute the corresponding best alignments with a traceback.

## Software requirements
* Linux operating system with compatible CUDA Toolkit 12 or newer
* C++17 compiler

## Hardware requirements
* CUDA-capable GPU with Volta architecture or newer. The code should also compile for older GPUs, i.e. Pascal and earlier, but was not tested on those.


## Download
`git clone --recurse-submodules git@github.com:fkallen/gpu-tRNA-mapper.git`


## Setup

In the top folder, execute the following commands to build `gpu-tRNA-mapper`

Step 1. Set up Parasail:
```
cd parasail-2.6.2
mkdir build
cd build
cmake ..
make
cd ../..
```

Alternatively, modify the Makefile to point to an existing parasail library in your system.
In any case, make sure that libparasail.so is added to `LD_LIBRARY_PATH`

Step 2. Build gpu-tRNA-mapper:
```
make gpu-tRNA-mapper [build-options]
```

Build options:  
- **GPUARCH=targetarch** : Specify the target GPU architecture
    - **GPUARCH=native** (DEFAULT) :  Compile code for all GPU archictectures of GPUs detected in the machine. The CUDA environment variable `CUDA_VISIBLE_DEVICES` can be used to control the detected GPUs. If `CUDA_VISIBLE_DEVICES` is not set, it will default to all GPUs in the machine.
    - **GPUARCH=all** : Compile code for all GPU architectures supported by the current CUDA toolkit, and ensure forward compatibility for unreleased GPU architectures
    - **GPUARCH=all-major** : Compile code for all major GPU architectures supported by the current CUDA toolkit, and ensure forward compatibility for unreleased GPU architectures
    - **GPUARCH=sm_XY** : Compile only for the single GPU architecture with major version X and minor version Y. For example, "GPUARCH=sm_89" to target the ADA architecture
- **GPUARCH_NUM_COMPILE_THREADS=N** : Parallelize compilation of multiple GPU architectures using N threads. (Default N = 1)


Step 3 (optional). Install gpu-tRNA-mapper
```
make install PREFIX=installdir
```

Copies the executable to directory "installdir/bin". Default is **PREFIX=/usr/local**



## Usage

Minimal example command: 
```
    ./gpu-tRNA-mapper --readFileName data/reads.fastq --referenceFileName data/trna_ref.fasta --outputFileName data/output.sam
```

Mandatory arguments:
```
    --readFileName file : fasta or fastq, can be .gz file
    --referenceFileName file : fasta or fastq, can be .gz file
    --outputFileName file.sam
```

Optional arguments:
```
    --scoring matchscore,mismatchscore,gapopenscore,gapextendscore : the alignment score parameters (default: --scoring 2,-1,-10,-1)
    --semiglobalAlignment : Perform a semi-global alignment instead of a local alignment
    --verbose : More console output
    --batchsize num : Align num reads in parallel (default: 100000)
    --minAlignmentScore num : the best observed alignment score for a read must be >= num, otherwise the read will be treated as unmapped (default: 0)
    --resultListSize num : If a read can be mapped to multiple reference sequences with the same best score, output up to num alignments (default: 2147483647 (output all best mappings))
```

Input files can be DNA or RNA. We support a four letter alphabet A, C, G, (T/U) , where T matches U and vice-versa.

gpu-tRNA-mapper uses GPU 0. Use the CUDA environment variable `CUDA_VISIBLE_DEVICES` to select the GPU in multi-GPU systems

gpu-tRNA-mapper makes use of CPU multi-threading with OpenMP. The environment variable `OMP_NUM_THREADS` is used to control the number of CPU threads.


Advanced example command:

```
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 ./gpu-tRNA-mapper --readFileName data/reads.fastq --referenceFileName data/trna_ref.fasta --outputFileName data/output.sam --scoring 5,-1,-5,-1 --minAlignmentScore 60 --resultListSize 10
```
