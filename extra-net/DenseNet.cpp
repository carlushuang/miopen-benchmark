#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


void DenseNet() {

    TensorDesc input_dim(32, 3, 224, 224);

    Sequential features(input_dim);

    /* features */

    features.addConv(16, 3, 1, 1);
    features.addReLU();
    features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(160, 1, 0, 1);
    features.addReLU();
    features.addMaxPool(2, 0, 2);


	features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(304, 1, 0, 1);
    features.addReLU();
	features.addMaxPool(2, 0, 2);


	features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addMaxPool(2, 0, 2);

    DEBUG("Dims after Features: " << features.getOutputDesc());

    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 12 *28 * 28,  1, 1);

    classifier.addLinear(10);

    classifier.addReLU();

  

    Model m(input_dim);

    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("dense_net");

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

    DenseNet();

#ifdef __NVCC__
    cudnnDestroy(cudnn::handle());
#else
    miopenDestroy(mio::handle());
#endif

    return 0;

}
