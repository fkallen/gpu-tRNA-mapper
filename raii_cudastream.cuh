#ifndef RAII_CUDASTREAM_CUH
#define RAII_CUDASTREAM_CUH

#include "cuda_errorcheck.cuh"

class CudaStream{
public:
    CudaStream(){
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaStreamCreate(&stream));
    }
    CudaStream(unsigned int flags){
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaStreamCreateWithFlags(&stream, flags));
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream(CudaStream&& rhs){
        destroy();
        deviceId = rhs.deviceId;
        stream = std::exchange(rhs.stream, nullptr);
    }

    ~CudaStream(){
        destroy();
    }

    void destroy(){
        if(stream != nullptr){
            int d;
            CUDACHECK(cudaGetDevice(&d));
            CUDACHECK(cudaSetDevice(deviceId));

            CUDACHECK(cudaStreamDestroy(stream));
            stream = nullptr;

            CUDACHECK(cudaSetDevice(d));
        }
    }

    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream& operator=(CudaStream&& rhs){
        swap(*this, rhs);

        return *this;
    }

    friend void swap(CudaStream& l, CudaStream& r) noexcept
    {
        std::swap(l.deviceId, r.deviceId);
        std::swap(l.stream, r.stream);
    }

    cudaError_t query() const{
        return cudaStreamQuery(stream);
    }

    cudaError_t synchronize() const{
        return cudaStreamSynchronize(stream);
    }

    cudaError_t waitEvent(cudaEvent_t event, unsigned int flags) const{
        return cudaStreamWaitEvent(stream, event, flags);
    }

    operator cudaStream_t() const{
        return stream;
    }

    int getDeviceId() const{
        return deviceId;
    }

    cudaStream_t getStream() const{
        return stream;
    }
private:

    int deviceId{};
    cudaStream_t stream{};
};


#endif