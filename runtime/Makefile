SRC_DIR     := ./src
OBJ_DIR     := ./obj
TARGET_DIR  := ./lib
TARGET      := $(TARGET_DIR)/libsnurhac.so

CXX_SRCS    := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp)
CXX_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CXX_SRCS))

CXX         := mpic++
CXXFLAGS    := -std=c++11 -Wall -O3
CPPFLAGS    := -I$(SRC_DIR) -I/usr/include/openmpi-x86_64 -I$(SNURHACCUDA)/include
LDFLAGS     := -L/usr/lib64/openmpi/lib -lmpi -L$(SNURHACCUDA)/lib64 -lcudart -lpthread
SOFLAGS     := -fPIC

NVCC        := nvcc
NVCC_FLAGS  :=

all: $(TARGET)

$(TARGET): $(CXX_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) -shared $^ -o $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOFLAGS) -c -o $@ $^

clean:
	rm -f $(CXX_OBJS)
	rm -f $(TARGET)
