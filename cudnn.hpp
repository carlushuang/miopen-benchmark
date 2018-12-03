#ifndef MY_CUDNN_HPP
#define MY_CUDNN_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <dirent.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <assert.h>

#define DEBUG(msg) std::cerr << "[DEBUG] " << msg << std::endl;
#define WARNING(msg) std::cerr << "[!WARNING!] " << msg << std::endl;
#define FATAL(msg) {std::cerr << "[!FATAL!] " << msg << std::endl; exit(EXIT_FAILURE);} while(0)
#define INFO(msg) std::cout << "[INFO]  " << msg << std::endl;

//#define FINISH() exit(EXIT_FAILURE);
#define FINISH()   assert(0)

#define CHECK_CUDA(cmd) \
{\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(cuda_error), cuda_error,__FILE__, __LINE__); \
        FINISH();\
    }\
}

#define CHECK_CUDRI(cmd) \
{\
    CUresult cu_error  = cmd;\
    if (cu_error != CUDA_SUCCESS) { \
        fprintf(stderr, "error: cu %d at %s:%d\n", cu_error,__FILE__, __LINE__); \
        FINISH();\
    }\
}

#define device_mem_t void*

device_mem_t device_alloc(size_t size) {
    void* ptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void device_free(device_mem_t m) {
    CHECK_CUDA(cudaFree(m));
}

std::vector<std::string> split(const std::string& str, char sep) {
    std::vector<std::string> strings;
    std::istringstream f(str);
    std::string s;
    while (std::getline(f, s, sep)) {
        strings.push_back(s);
    }
    return strings;
}

void print_file(const std::string& fname) {
    std::ifstream f(fname);
    std::string s;
    std::cout << "File: " << fname << ", status: " << f.good() << std::endl;
    while (std::getline(f, s)) {
        std::cout << ":: " << s << std::endl;
    }

}

std::vector<std::string> ls_dir(const std::string& dname) {
    std::vector<std::string> files;
    struct dirent* entry;
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        return files;
    }

    while ((entry = readdir(dir)) != NULL) {
        std::string fname(entry->d_name);
        if (fname != "." && fname != "..")
            files.push_back(fname);
    }
    return files;
}

std::vector<std::string> ls_dir(const std::string& dname, const std::regex& match) {
    std::vector<std::string> files;
    struct dirent* entry;
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        return files;
    }

    while ((entry = readdir(dir)) != NULL) {
        std::string fname(entry->d_name);
        if (fname != "." && fname != "..") {
            if (std::regex_match(fname, match)) {
                files.push_back(fname);
            }
        }
    }
    return files;
}

int read_current_mhz(const std::string& fname) {
    std::ifstream f(fname);
    std::string line;
    while (std::getline(f, line)) {
        if (line.back() == '*') {
            std::string mhzstr = line.substr(3, line.size()-3-5);
            std::istringstream iss(mhzstr);
            int mhz;
            iss >> mhz;
            return mhz;
        }
    }
    return -1;
}

typedef struct
{
    char bus_id_str[16]; /* string form of bus info */
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pci_device_id; /* combined device and vendor id */
    unsigned int pci_subsystem_id;
    unsigned int res0; /* NVML internal use only */
    unsigned int res1;
    unsigned int res2;
    unsigned int res3;
} wrap_nvmlPciInfo_t;
typedef enum wrap_nvmlReturn_enum { WRAPNVML_SUCCESS = 0 } wrap_nvmlReturn_t;
typedef void* wrap_nvmlDevice_t;
typedef struct _wrap_nvml_handle
{
    void* nvml_dll;
    int nvml_gpucount;
    unsigned int* nvml_pci_domain_id;
    unsigned int* nvml_pci_bus_id;
    unsigned int* nvml_pci_device_id;
    wrap_nvmlDevice_t* devs;
    wrap_nvmlReturn_t (*nvmlInit)(void);
    wrap_nvmlReturn_t (*nvmlDeviceGetCount)(int*);
    wrap_nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, wrap_nvmlDevice_t*);
    wrap_nvmlReturn_t (*nvmlDeviceGetPciInfo)(wrap_nvmlDevice_t, wrap_nvmlPciInfo_t*);
    wrap_nvmlReturn_t (*nvmlDeviceGetName)(wrap_nvmlDevice_t, char*, int);
    wrap_nvmlReturn_t (*nvmlDeviceGetTemperature)(wrap_nvmlDevice_t, int, unsigned int*);
    wrap_nvmlReturn_t (*nvmlDeviceGetFanSpeed)(wrap_nvmlDevice_t, unsigned int*);
    wrap_nvmlReturn_t (*nvmlDeviceGetPowerUsage)(wrap_nvmlDevice_t, unsigned int*);
    wrap_nvmlReturn_t (*nvmlShutdown)(void);

    _wrap_nvml_handle(){nvml_dll=nullptr;}
    ~_wrap_nvml_handle(){
        if(nvml_dll){
            dlclose(nvml_dll);

            free(devs);
            free(nvml_pci_domain_id);
            free(nvml_pci_bus_id);
            free(nvml_pci_device_id);
        }
    }
} wrap_nvml_handle;

struct Device {
    int gpu_id;
    cudaDeviceProp props;
    CUdevice cu_id;
    CUcontext cu_ctx;
    CUstream  cu_stream;

    void print_info();
    float getTemp();
    int getFanspeed();
    int getClock();
    int getMemClock();

    void init_cuda_driver();
    void finish_cuda_driver();
    ~Device(){
        finish_cuda_driver();
    }
};

struct Devices {
    static wrap_nvml_handle * nvml_handle(){
        static wrap_nvml_handle nvml_handle_;
        return &nvml_handle_;
    }
    static void init_nv_hwmon(){
#define NVIDIA_ML_LIB  "libnvidia-ml.so"
        wrap_nvml_handle* nvmlh = nvml_handle();
        void* nvml_dll = dlopen(NVIDIA_ML_LIB, RTLD_NOW);
        if (nvml_dll == nullptr){
            WARNING("can't load "NVIDIA_ML_LIB);
            return;
        }

        //nvmlh = (wrap_nvml_handle*)calloc(1, sizeof(wrap_nvml_handle));

        nvmlh->nvml_dll = nvml_dll;

        nvmlh->nvmlInit = (wrap_nvmlReturn_t(*)(void))dlsym(nvmlh->nvml_dll, "nvmlInit");
        nvmlh->nvmlDeviceGetCount =
            (wrap_nvmlReturn_t(*)(int*))dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
        nvmlh->nvmlDeviceGetHandleByIndex = (wrap_nvmlReturn_t(*)(int, wrap_nvmlDevice_t*))dlsym(
            nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
        nvmlh->nvmlDeviceGetPciInfo = (wrap_nvmlReturn_t(*)(
            wrap_nvmlDevice_t, wrap_nvmlPciInfo_t*))dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
        nvmlh->nvmlDeviceGetName = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, char*, int))dlsym(
            nvmlh->nvml_dll, "nvmlDeviceGetName");
        nvmlh->nvmlDeviceGetTemperature = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, int,
            unsigned int*))dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
        nvmlh->nvmlDeviceGetFanSpeed = (wrap_nvmlReturn_t(*)(
            wrap_nvmlDevice_t, unsigned int*))dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
        nvmlh->nvmlDeviceGetPowerUsage = (wrap_nvmlReturn_t(*)(
            wrap_nvmlDevice_t, unsigned int*))dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
        nvmlh->nvmlShutdown = (wrap_nvmlReturn_t(*)())dlsym(nvmlh->nvml_dll, "nvmlShutdown");
        if (nvmlh->nvmlInit == nullptr || nvmlh->nvmlShutdown == nullptr ||
            nvmlh->nvmlDeviceGetCount == nullptr || nvmlh->nvmlDeviceGetHandleByIndex == nullptr ||
            nvmlh->nvmlDeviceGetPciInfo == nullptr || nvmlh->nvmlDeviceGetName == nullptr ||
            nvmlh->nvmlDeviceGetTemperature == nullptr || nvmlh->nvmlDeviceGetFanSpeed == nullptr ||
            nvmlh->nvmlDeviceGetPowerUsage == nullptr)
        {
            WARNING(NVIDIA_ML_LIB " can't find some symbol");
            dlclose(nvmlh->nvml_dll);
            nvmlh->nvml_dll = nullptr;
            return;
        }

        nvmlh->nvmlInit();
        nvmlh->nvmlDeviceGetCount(&nvmlh->nvml_gpucount);

        nvmlh->devs = (wrap_nvmlDevice_t*)calloc(nvmlh->nvml_gpucount, sizeof(wrap_nvmlDevice_t));
        nvmlh->nvml_pci_domain_id = (unsigned int*)calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
        nvmlh->nvml_pci_bus_id = (unsigned int*)calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
        nvmlh->nvml_pci_device_id = (unsigned int*)calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));

        /* Obtain GPU device handles we're going to need repeatedly... */
        for (int i = 0; i < nvmlh->nvml_gpucount; i++)
        {
            nvmlh->nvmlDeviceGetHandleByIndex(i, &nvmlh->devs[i]);
        }

        /* Query PCI info for each NVML device, and build table for mapping of */
        /* CUDA device IDs to NVML device IDs and vice versa                   */
        for (int i = 0; i < nvmlh->nvml_gpucount; i++)
        {
            wrap_nvmlPciInfo_t pciinfo;
            nvmlh->nvmlDeviceGetPciInfo(nvmlh->devs[i], &pciinfo);
            nvmlh->nvml_pci_domain_id[i] = pciinfo.domain;
            nvmlh->nvml_pci_bus_id[i] = pciinfo.bus;
            nvmlh->nvml_pci_device_id[i] = pciinfo.device;
        }

        //return nvmlh;
    }

    static std::vector<Device>& get_devices(bool from_init = false) {
        static bool is_init = false;
        static std::vector<Device> d;
        if (!is_init) {
            is_init = true;
            if (!from_init)
                init_devices();
        }
        return d;
    }

    static Device& get_default_device() {
        if (get_devices().size() == 0) {
            FATAL("No HIP Devices available.");
        }
        return get_devices()[0];
    }

    static void init_devices() {
        int devcount;
        CHECK_CUDA(cudaGetDeviceCount(&devcount));
        INFO("Number of CUDA devices found: " << devcount);

        if (devcount == 0) {
            FATAL("No CUDA devices found.");
        }

        std::vector<Device>& devs = get_devices(true);
        devs.resize(devcount);

        // init and get devices
        for (int d = 0; d < devcount; ++d) {
            devs[d].gpu_id = d;
            CHECK_CUDA(cudaGetDeviceProperties(&devs[d].props, d/*deviceID*/));
            //devs[d].init_sys_paths();
            devs[d].init_cuda_driver();
            devs[d].print_info();
        }
    }

};

// docs.nvidia.com/cuda/cuda-driver-api/index.html
void Device::init_cuda_driver(){
    CHECK_CUDRI(cuDeviceGet(&cu_id, gpu_id));
    CHECK_CUDRI(cuDevicePrimaryCtxRetain(&cu_ctx, cu_id));
    CHECK_CUDRI(cuCtxSetCurrent(cu_ctx));
    CHECK_CUDRI(cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT));
}
void Device::finish_cuda_driver(){
    cuDevicePrimaryCtxRelease(cu_id);
    cuStreamDestroy(cu_stream);
}

void Device::print_info() {
    // print out device info
    INFO("Device " << gpu_id << ": " << props.name);
    INFO("\tArch:\t" << props.major<<props.minor);  // cuda compute capability
    INFO("\tGMem:\t" << props.totalGlobalMem/1024/1024 << " MiB");
    INFO("\twarps:\t" << props.warpSize);
    INFO("\tCUs:\t" << props.multiProcessorCount);
    INFO("\tMaxClk:\t" << props.clockRate);
    INFO("\tMemClk:\t" << props.memoryClockRate);
    //INFO("\tdrm:\t" << drm_path);
    //INFO("\thwmon:\t" << hwmon_path);
    //INFO("\t\tpciDomainID:\t" << props.pciDomainID);
    //INFO("\t\tpciBusID:\t" << props.pciBusID);
    //INFO("\t\tpciDeviceID:\t" << props.pciDeviceID);

}

float Device::getTemp() {
    wrap_nvml_handle* nvmlh = Devices::nvml_handle();
    if(!nvmlh->nvml_dll)
        return -1;
    unsigned int tempC;
    if (nvmlh->nvmlDeviceGetTemperature(
        nvmlh->devs[gpu_id], 0u /* NVML_TEMPERATURE_GPU */, &tempC) != WRAPNVML_SUCCESS)
        return -1;
    return (float)tempC;
}

int Device::getFanspeed() {
    wrap_nvml_handle* nvmlh = Devices::nvml_handle();
    if(!nvmlh->nvml_dll)
        return -1;
    unsigned int fanpcnt;
    if (nvmlh->nvmlDeviceGetFanSpeed(nvmlh->devs[gpu_id], &fanpcnt) != WRAPNVML_SUCCESS)
        return -1;
    return fanpcnt;
}

int Device::getClock() {
    // TODO: fixme
    return props.clockRate;
}

int Device::getMemClock() {
    // TODO: fixme
    return props.memoryClockRate;
}


void device_init() {
    Devices::init_devices();
}

#define CHECK_CUDNN(cmd) \
{\
    cudnnStatus_t stat = cmd;\
    if (stat != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudnnGetErrorString(stat), stat,__FILE__, __LINE__); \
        FINISH();\
    }\
}

// get cudnnHandle globally via `cudnn::handle()`
struct cudnn {
private:
    // This is called once, the first time the MIOpen handle is retrieved
    static cudnnHandle_t init_cudnn() {
        cudnnHandle_t h;
        CHECK_CUDA(cudaSetDevice(Devices::get_default_device().gpu_id));
        //cudnnStream_t q;
        //CHECK_CUDA(cudaStreamCreate(&q));
        CHECK_CUDNN(cudnnCreate(&h));
        return h;
    }
public:
    static cudnnHandle_t handle() {
        static cudnnHandle_t h = init_cudnn();
        return h;
    }
};


float getTemp() {
    return Devices::get_default_device().getTemp();
}

int getFanspeed() {
    return Devices::get_default_device().getFanspeed();
}

int getClock() {
    return Devices::get_default_device().getClock();
}

int getMemClock() {
    return Devices::get_default_device().getMemClock();
}

// utils function
void dump_cudnn_struct(cudnnTensorDescriptor_t x){
    cudnnDataType_t type;
    int nd;
    std::vector<int> dims(4);
    std::vector<int> strides(4);
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(x, 4, &type, &nd, dims.data(), strides.data()));
    std::cout<<"cudnnTensorDescriptor_t: dtype:"<<type<<", nd:"<<nd;
    std::cout<<", dim:[";
    for(auto d : dims)
        std::cout<<d<<",";
    std::cout<<"]";
    std::cout<<", strides:[";
    for(auto s : strides)
        std::cout<<s<<",";
    std::cout<<"]"<<std::endl;
}

void dump_cudnn_struct(cudnnFilterDescriptor_t x){
    cudnnDataType_t type;
    int nd;
    cudnnTensorFormat_t format;
    std::vector<int> kernel(4);
    CHECK_CUDNN(cudnnGetFilterNdDescriptor(x, 4, &type, &format, &nd, kernel.data()));
    std::cout<<"cudnnFilterDescriptor_t: dtype:"<<type<<", fmt:"<<format<<", nd:"<<nd;
    std::cout<<", kernel:[";
    for(auto k : kernel)
        std::cout<<k<<",";
    std::cout<<"]"<<std::endl;
}

void dump_cudnn_struct(cudnnConvolutionDescriptor_t x){
    cudnnDataType_t type;
    cudnnConvolutionMode_t mode;
    int nd;
    std::vector<int> pads(3);
    std::vector<int> strides(3);
    std::vector<int> dilations(3);
    CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(
      x, 3, &nd, pads.data(), strides.data(), dilations.data(), &mode,
      &type));
    std::cout<<"cudnnConvolutionDescriptor_t: dtype:"<<type<<", conv_mode:"<<mode<<", nd:"<<nd;
    std::cout<<", pads:[";
    for(auto i : pads)
        std::cout<<i<<",";
    std::cout<<"]";
    std::cout<<", strides:[";
    for(auto i : strides)
        std::cout<<i<<",";
    std::cout<<"]";
    std::cout<<", dilations:[";
    for(auto i : dilations)
        std::cout<<i<<",";
    std::cout<<"]"<<std::endl;
}

#endif
