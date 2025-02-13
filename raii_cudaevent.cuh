#ifndef RAII_CUDAEVENT_CUH
#define RAII_CUDAEVENT_CUH

#include "cuda_errorcheck.cuh"

class CudaEvent{
public:
    CudaEvent(){
        CUDACHECK(cudaGetDevice(&deviceId)); 
        CUDACHECK(cudaEventCreate(&event)); 
    }
    CudaEvent(unsigned int flags){
        CUDACHECK(cudaGetDevice(&deviceId)); 
        CUDACHECK(cudaEventCreateWithFlags(&event, flags)); 
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& rhs){
        destroy();
        deviceId = rhs.deviceId;
        event = std::exchange(rhs.event, nullptr);
    }

    ~CudaEvent(){
        destroy();
    }

    void destroy(){
        if(event != nullptr){
            int d;
            CUDACHECK(cudaGetDevice(&d)); 
            CUDACHECK(cudaSetDevice(deviceId)); 

            CUDACHECK(cudaEventDestroy(event)); 
            event = nullptr;

            CUDACHECK(cudaSetDevice(d)); 
        }
    }

    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent& operator=(CudaEvent&& rhs){
        swap(*this, rhs);

        return *this;
    }

    friend void swap(CudaEvent& l, CudaEvent& r) noexcept
    {
        std::swap(l.deviceId, r.deviceId);
        std::swap(l.event, r.event);
    }

    cudaError_t query() const{
        return cudaEventQuery(event);
    }

    cudaError_t record(cudaStream_t stream = 0) const{
        return cudaEventRecord(event, stream);
    }

    cudaError_t synchronize() const{
        return cudaEventSynchronize(event);
    }

    cudaError_t elapsedTime(float* ms, cudaEvent_t end) const{
        return cudaEventElapsedTime(ms, event, end);
    }

    operator cudaEvent_t() const{
        return event;
    }

    int getDeviceId() const{
        return deviceId;
    }

    cudaEvent_t getEvent() const{
        return event;
    }
private:

    int deviceId{};
    cudaEvent_t event{};
};


#endif