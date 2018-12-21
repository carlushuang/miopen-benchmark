ROCM_PATH?= $(wildcard /opt/rocm)
HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc
INCLUDE_DIRS=-I$(HIP_PATH)/include -I$(ROCM_PATH)/include -I$(ROCM_PATH)/hipblas/include -I./
LD_FLAGS=-L$(ROCM_PATH)/lib -L$(ROCM_PATH)/opencl/lib/x86_64 -lMIOpen -lOpenCL -lmiopengemm -lhipblas -lrocblas
TARGET=--amdgpu-target=gfx900 --amdgpu-target=gfx906
LAYER_TIMING=1

#HIPCC_FLAGS=-g -Wall $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)
HIPCC_FLAGS=-g -O3 -Wall -DLAYER_TIMING=$(LAYER_TIMING) $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)


all: alexnet resnet benchmark_wino layerwise gputop extra_net

HEADERS=function.hpp layers.hpp miopen.hpp multi_layers.hpp tensor.hpp utils.hpp

benchmark: all
	./benchmark_wino W1 1000 | tee W1.log \
	&& ./benchmark_wino L2 10000 | tee L2.log \
	&& ./layerwise | tee layerwise.log \
	&& ./alexnet | tee alexnet.log \
	&& ./resnet | tee resnet50.log

alexnet: alexnet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) alexnet.cpp $(LD_FLAGS) -o $@

main: main.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) main.cpp $(LD_FLAGS) -o $@

gputop: gputop.cpp miopen.hpp
	$(HIPCC) $(HIPCC_FLAGS) gputop.cpp $(LD_FLAGS) -o $@

resnet: resnet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) resnet.cpp $(LD_FLAGS) -o $@

benchmark_wino: benchmark_wino.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) benchmark_wino.cpp $(LD_FLAGS) -o $@

layerwise: layerwise.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) layerwise.cpp $(LD_FLAGS) -o $@

extra_net: extra-net/vgg16 extra-net/vgg19 extra-net/shufflenet extra-net/resnext extra-net/mobilenet \
	extra-net/mobilenet extra-net/mobilenet_v2 extra-net/inception_v3 extra-net/inception_v4 \
	extra-net/googlenet extra-net/densenet

extra-net/vgg16: extra-net/VGG16.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/VGG16.cpp $(LD_FLAGS) -o $@

extra-net/vgg19: extra-net/VGG19.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/VGG19.cpp $(LD_FLAGS) -o $@

extra-net/shufflenet: extra-net/ShuffleNet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/ShuffleNet.cpp $(LD_FLAGS) -o $@

extra-net/resnext: extra-net/resneXt.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/resneXt.cpp $(LD_FLAGS) -o $@

extra-net/mobilenet: extra-net/MobileNet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/MobileNet.cpp $(LD_FLAGS) -o $@

extra-net/mobilenet_v2: extra-net/MobileNet_v2.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/MobileNet_v2.cpp $(LD_FLAGS) -o $@

extra-net/inception_v3: extra-net/Inception_v3.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/Inception_v3.cpp $(LD_FLAGS) -o $@

extra-net/inception_v4: extra-net/Inception_v4.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/Inception_v4.cpp $(LD_FLAGS) -o $@

extra-net/googlenet: extra-net/GoogleNet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/GoogleNet.cpp $(LD_FLAGS) -o $@

extra-net/densenet: extra-net/DenseNet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) extra-net/DenseNet.cpp $(LD_FLAGS) -o $@

clean:
	rm -f *.o *.out benchmark segfault alexnet resnet benchmark_wino layerwise gputop main \
	extra-net/vgg16 extra-net/vgg19 extra-net/shufflenet extra-net/resnext extra-net/mobilenet \
	extra-net/mobilenet extra-net/mobilenet_v2 extra-net/inception_v3 extra-net/inception_v4 \
	extra-net/googlenet extra-net/densenet
