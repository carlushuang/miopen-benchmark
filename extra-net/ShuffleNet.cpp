#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"



void ShuffleNet() {

    TensorDesc input_dim(128, 3, 224, 224);

    Sequential features(input_dim);


    /* features */

	features.addConv(24, 3, 1, 2);
	features.addReLU();

	features.addMaxPool(3, 0, 2);

	features.addConv(54, 1, 0, 1);
	features.addReLU();

	features.addConv(54, 3, 1, 2);
	features.addReLU();

	features.addConv(216, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 3, 1, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 3, 1, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 3, 1, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addMaxPool(3, 0, 2);

	features.addConv(60, 1, 0, 1);
	features.addReLU();

	features.addConv(60, 3, 1, 2);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 1);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addMaxPool(3, 0, 2);

	features.addConv(120, 1, 0, 1);
	features.addReLU();

	features.addConv(120, 3, 1, 2);
	features.addReLU();

	features.addConv(480, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 3, 1, 1);
	features.addReLU();

	features.addConv(960, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 3, 1, 1);
	features.addReLU();

	features.addConv(960, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 1, 0, 1);
	features.addReLU();

	features.addConv(240, 3, 1, 1);
	features.addReLU();

	features.addConv(960, 1, 0, 1);
	features.addReLU();

	features.addMaxPool(1, 0, 1);

	features.addConv(1000, 1, 0, 1);


   Model m(input_dim);
	
   m.add(features);


   BenchmarkLogger::new_session("shuffle_net");

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

    ShuffleNet();

#ifdef __NVCC__
    cudnnDestroy(cudnn::handle());
#else
    miopenDestroy(mio::handle());
#endif

    return 0;

}