#ifdef __NVCC__
#include "cudnn.hpp"
#else
#include "miopen.hpp"
#endif
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


// implements AlexNet
void alexNet() {
    TensorDesc input_dim(128, 3, 224, 224);

    Sequential features(input_dim);
    /* features */
    features.addConv(64, 11, 2, 4);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(192, 5, 2, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(384, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);

    DEBUG("Dims after Features: " << features.getOutputDesc());

    /* classifier */
    Sequential classifier(features.getOutputDesc());
    // TODO Dropout
    classifier.reshape(input_dim.n, 256 * 6 * 6, 1, 1);
    classifier.addLinear(4096);
    classifier.addReLU();
    // TODO: Dropout
    classifier.addLinear(4096);
    classifier.addReLU();
    classifier.addLinear(1000);

    Model m(input_dim);
    m.add(features);
    m.add(classifier);
    m.input.uniform(); // randomly initialize input

    BenchmarkLogger::new_session("alex_net");
    BenchmarkLogger::benchmark(m, 50);
}


int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
#ifdef __NVCC__
#else
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));
#endif

    alexNet();
#ifdef __NVCC__
    cudnnDestroy(cudnn::handle());
#else
    miopenDestroy(mio::handle());
#endif
    return 0;
}
