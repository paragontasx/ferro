CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fno-math-errno -fPIC -MMD -MP
LDFLAGS = -pthread -ldl
CL_INC ?= ../ferro-gpu/third_party/OpenCL-Headers
override CPPFLAGS += -I$(CL_INC)

SRC = src/gpu_cl.cpp src/gguf.cpp src/kernels.cpp src/kernels_fma.cpp src/threadpool.cpp src/model_eg2.cpp src/server.cpp src/tokenizer.cpp \
      src/tokenizer_bpe.cpp src/json.cpp src/decision.cpp src/model_lfm2.cpp \
      src/media.cpp src/media_impl.cpp src/mm_preproc.cpp src/mm_encoder.cpp src/mm_gemma4.cpp src/mm_gemma4a.cpp src/embed_mm.cpp \
      third_party/llama_unicode/unicode.cpp third_party/llama_unicode/unicode-data.cpp
OBJ = $(SRC:.cpp=.o)

all: ferro-embed

ferro-embed: $(OBJ) src/main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $(OBJ) src/main.cpp $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

test: ferro-embed tests/test_kernels tests/test_decision tests/test_media
	./tests/test_kernels
	./tests/test_decision
	./tests/test_media

tests/test_decision: tests/test_decision.cpp src/json.o src/decision.o src/media.o src/media_impl.o src/tokenizer_bpe.o src/gguf.o third_party/llama_unicode/unicode.o third_party/llama_unicode/unicode-data.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -Isrc -o $@ $^ $(LDFLAGS)

tests/test_media: tests/test_media.cpp src/media.o src/media_impl.o src/mm_preproc.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -Isrc -o $@ $^ $(LDFLAGS)

tests/test_kernels: tests/test_kernels.cpp $(filter-out src/main.cpp,$(SRC))
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -Isrc -o $@ tests/test_kernels.cpp $(filter-out src/main.cpp,$(SRC)) $(LDFLAGS)

-include $(OBJ:.o=.d)

clean:
	rm -f src/*.o src/*.d third_party/llama_unicode/*.o third_party/llama_unicode/*.d ferro-embed tests/test_kernels tests/test_decision tests/test_media
