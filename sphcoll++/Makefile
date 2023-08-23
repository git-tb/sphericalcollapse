EXE := "SPECIFY_EXECUTABLE" # make ... EXE=<your executable>

CXX       := g++
CXX_FLAGS := -std=c++2a
# CXX_FLAGS += -g
CXX_FLAGS += -O2
CXX_FLAGS += -DNDEBUG
CXX_FLAGS += -fopenmp

BIN     := bin
SRC     := src

INCLUDE :=
DIRLIBRARIES := 
LIBRARIES :=

INCLUDE += -I./include
# INCLUDE += -I./clipp-master/include
INCLUDE += -I./eigen-3.4.0/Eigen
# INCLUDE += -I./nlopt-2.7.1/build/install/include

# DIRLIBRARIES += -L./nlopt-2.7.1/build/install/lib

# LIBRARIES += -lopenblas
# LIBRARIES += -lgsl
# LIBRARIES += -lm
# LIBRARIES += -lnlopt

all: build

download:
	wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
	tar -xvzf eigen-3.4.0.tar.gz

	wget https://github.com/stevengj/nlopt/archive/v2.7.1.tar.gz
	tar -xvzf v2.7.1.tar.gz

	wget https://github.com/muellan/clipp/archive/refs/heads/master.zip
	unzip master.zip

build:
	$(CXX) $(CXX_FLAGS) $(INCLUDE) $(SRC)/$(EXE).cpp -o $(BIN)/$(EXE) $(DIRLIBRARIES) $(LIBRARIES) 
#-Wl,-rpath=$(PWD)/nlopt-2.7.1/build/install/lib

checklibs:
	ldd ./bin/$(EXE)
