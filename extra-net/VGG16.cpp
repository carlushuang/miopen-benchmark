#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"



void VGG16() {

    TensorDesc input_dim(32, 3, 224, 224);

    Sequential features(input_dim);

    /* features */

	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 512 * 7 * 7, 1, 1);

    classifier.addLinear(4096);

    classifier.addReLU();

    // TODO: Dropout

    classifier.addLinear(4096);

    classifier.addReLU();

    classifier.addLinear(1000);


    Model m(input_dim);

    m.add(features);

    m.add(classifier);


    BenchmarkLogger::new_session("VGG_16");

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

    VGG16();

#ifdef __NVCC__
    cudnnDestroy(cudnn::handle());
#else
    miopenDestroy(mio::handle());
#endif

    return 0;

}
