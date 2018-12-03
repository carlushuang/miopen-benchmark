

#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include <chrono>
#include <thread>


int main(int argc, char *argv[])
{
    INFO("Finding & Initializing Devices...");
    std::vector<Device> devs = Devices::get_devices();

    while (1) {
        std::cout << "GPU\t\t\tTemp\tFan\tClk\t\tMClk" << std::endl;
        for (size_t i = 0; i < devs.size(); ++i) {
#ifdef __NVCC__
            std::cout << "GPU " << devs[i].gpu_id << " (" << devs[i].props.name << "):\t";
#else
            std::cout << "GPU " << devs[i].hip_id << " (" << devs[i].hip_props.name << "):\t";
#endif
            std::cout << devs[i].getTemp() << " C\t";
            std::cout << devs[i].getFanspeed()*100/255 << " %\t";
            std::cout << devs[i].getClock() << " MHz\t";
            std::cout << devs[i].getMemClock() << " MHz\t";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
