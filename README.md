# gpu-tRNA-mapper
gpu-tRNA-mapper: GPU-accelerated read mapper to map against short tRNA sequences.


For a set of input reads and reference sequences, gpu-tRNA-mapper aligns each read to all reference sequences.
By default, an optimal local alignment is used. 
For each read, the highest scoring alignment is written to output file in SAM format. If multiple references share the same highest score all alignments are written to output file.

The computation of the all-to-all alignment scores is GPU-accelerated. After determining the highest alignment score for each read, the parasail alignment library is used on the CPU to recompute the corresponding best alignments with a traceback.

## Software requirements

* Linux operating system with compatible CUDA Toolkit 12 or newer
* C++17 compiler
* parasail: https://github.com/jeffdaily/parasail - tested with version 2.6.2

By default parasail-2.6.2 will be searched in the local directory. So you may
compile it there like this:

```
cd parasail-2.6.2
mkdir build
cd build
cmake ..
make
cd ../..
```

## Hardware requirements
* CUDA-capable GPU with Volta architecture or newer. The code should also compile for older GPUs, i.e. Pascal and earlier, but was not tested on those.


## Download
`git clone --recurse-submodules git@github.com:fkallen/gpu-tRNA-mapper.git`


## External dependencies

We require a correctly set up Parasail 2.6.2 alignment library (https://github.com/jeffdaily/parasail).

## Setup

Step 1. Build gpu-tRNA-mapper:
```
make -j [build-options]
```

You may require one or more of the following build options:

- **PARASAIL_INCLUDE_DIR** : Parasail's include files
    - Default: parasail-2.6.2/include
    - Example: /usr/local/parasail/include
- **PARASAIL_LIB_DIR** : Paradails's library path - should contain a libparasail.so
    - Default: parasail-2.6.2
    - Example: /usr/local/parasail/lib
- **TARGET_GPU_ARCH=targetarch** : Specify the target GPU architecture(s) - set if you want to use the same binary with diffrent GPUs
    - **TARGET_GPU_ARCH=native** (DEFAULT) :  Compile code for all GPU archictectures of GPUs detected in the machine. The CUDA environment variable `CUDA_VISIBLE_DEVICES` can be used to control the detected GPUs. If `CUDA_VISIBLE_DEVICES` is not set, it will default to all GPUs in the machine.
    - **TARGET_GPU_ARCH=all** : Compile code for all GPU architectures supported by the current CUDA toolkit, and ensure forward compatibility for unreleased GPU architectures
    - **TARGET_GPU_ARCH=all-major** : Compile code for all major GPU architectures supported by the current CUDA toolkit, and ensure forward compatibility for unreleased GPU architectures
    - **TARGET_GPU_ARCH="XY"** : Compile only for the GPU architecture with major version X and minor version Y. Multiple XY can be specified separated by space. For example, TARGET_GPU_ARCH="89 90" compiles for both the Ada and Hopper architecture.
- **GPU_COMPILE_THREADS=N** : Parallelize compilation of multiple GPU architectures using N threads. (Default N = 1)

Step 2. (optional) Install gpu-tRNA-mapper
```
make install PREFIX=installdir
```

Copies the executable to directory "installdir/bin". Default is **PREFIX=/usr/local**

Please note, that this will *not* take care of Parasail. You may want to install
Parasail separately into a suitable location first. Next, link the binary again using
a suitable **PARASAIL_LIB_DIR** parameter, e.g.:
```
    cd parasail-2.6.2
    sudo make install
    cd ..
    rm gpu-tRNA-mapper
    make PARASAIL_INCLUDE_DIR=/usr/local/include PARASAIL_LIB_DIR=/usr/local/lib
    sudo make install
```
Without that you may have to set **LD_LIBRARY_PATH** to Parasail's lib folder to
ensure that the library will be found.

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
