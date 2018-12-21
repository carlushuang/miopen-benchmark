#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


void MobileNet_v2() {

    TensorDesc input_dim(16, 3, 224, 224);
	
    Sequential features(input_dim);

    /* features */

	features.addConv(32, 3, 1, 2);
	features.addReLU();

	features.addConv(32, 1, 1, 1);
	features.addReLU();

	features.addConv(32, 3, 1, 1);
	features.addReLU();

	features.addConv(16, 1, 1, 1);
	features.addReLU();

	features.addConv(96, 1, 1, 1);
	features.addReLU();

	features.addConv(96, 3, 1, 2);
	features.addReLU();

	features.addConv(24, 1, 1, 1);
	features.addReLU();

	features.addConv(144, 1, 1, 1);
	features.addReLU();

	features.addConv(144, 3, 1, 1);
	features.addReLU();

	features.addConv(24, 1, 1, 1);
	features.addReLU();

	features.addConv(144, 1, 1, 1);
	features.addReLU();

	features.addConv(144, 3, 1, 2);
	features.addReLU();

	features.addConv(32, 1, 1, 1);
	features.addReLU();

	features.addConv(192, 1, 1, 1);
	features.addReLU();

	features.addConv(192, 3, 1, 1);
	features.addReLU();

	features.addConv(32, 1, 1, 1);
	features.addReLU();

	features.addConv(192, 1, 1, 1);
	features.addReLU();

	features.addConv(192, 3, 1, 1);
	features.addReLU();

	features.addConv(32, 1, 1, 1);
	features.addReLU();

	features.addConv(192, 1, 1, 1);
    features.addReLU();

	features.addConv(192, 3, 1, 1);
    features.addReLU();

	features.addConv(64, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 3, 1, 1);
    features.addReLU();

	features.addConv(64, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 3, 1, 1);
    features.addReLU();

	features.addConv(64, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 3, 1, 1);
    features.addReLU();

	features.addConv(64, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 1, 1, 1);
    features.addReLU();

	features.addConv(384, 3, 1, 2);
	features.addReLU();

	features.addConv(96, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 3, 1, 1);
    features.addReLU();

	features.addConv(96, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 3, 1, 1);
    features.addReLU();

	features.addConv(96, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 1, 1, 1);
    features.addReLU();

	features.addConv(576, 3, 1, 2);
    features.addReLU();

	features.addConv(160, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 3, 1, 1);
    features.addReLU();

	features.addConv(160, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 3, 1, 1);
    features.addReLU();

	features.addConv(160, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 1, 1, 1);
    features.addReLU();

	features.addConv(960, 3, 1, 1);
    features.addReLU();

	features.addConv(320, 1, 1, 1);
    features.addReLU();

	features.addConv(1280, 1, 1, 1);
    features.addReLU();

	features.addMaxPool(1, 0, 1);

    features.addConv(1000, 1, 0, 1);


    Model m(input_dim);

    m.add(features);


    BenchmarkLogger::new_session("mobile_net_v2");

    BenchmarkLogger::benchmark(m, 1000);

}





int main(int argc, char *argv[])

{
    device_init();

    // enable profiling
#ifdef __NVCC__
#else
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));
#endif

    MobileNet_v2();

#ifdef __NVCC__
    cudnnDestroy(cudnn::handle());
#else
    miopenDestroy(mio::handle());
#endif

    return 0;
}


