#ifndef GRID_CONSTANT_HELPER
#define GRID_CONSTANT_HELPER

#if __CUDA_ARCH__ >= 700
#define GRID_CONSTANT_SPECIFIER __grid_constant__
#else
#define GRID_CONSTANT_SPECIFIER
#endif




#endif