CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_50


all: timer.o argument_parsing.o graph.o subgraph.o partitioner.o subgraph_generator.o gpu_kernels.o egraph_utilities.o

timer.o: timer.cpp
	$(CC) -c timer.cpp -o timer.o $(CFLAGS)
	
argument_parsing.o: argument_parsing.cu
	$(NC) -c argument_parsing.cu -o argument_parsing.o $(CFLAGS) $(NFLAGS)

graph.o: graph.cu
	$(NC) -c graph.cu -o graph.o $(CFLAGS) $(NFLAGS)

subgraph.o: subgraph.cu
	$(NC) -c subgraph.cu -o subgraph.o $(CFLAGS) $(NFLAGS)

partitioner.o: partitioner.cu
	$(NC) -c partitioner.cu -o partitioner.o $(CFLAGS) $(NFLAGS)

subgraph_generator.o: subgraph_generator.cu
	$(NC) -c subgraph_generator.cu -o subgraph_generator.o $(CFLAGS) $(NFLAGS)
	
gpu_kernels.o: gpu_kernels.cu
	$(NC) -c gpu_kernels.cu -o gpu_kernels.o $(CFLAGS) $(NFLAGS)

egraph_utilities.o: egraph_utilities.cpp
	$(CC) -c egraph_utilities.cpp -o egraph_utilities.o $(CFLAGS)

clean:
	rm *.o
