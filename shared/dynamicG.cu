#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>

template <class E>
dynamicGraph<E>::dynamicGraph(string graphFilePath, bool isWeighted)
{
	this->dynamicgraphFilePath = graphFilePath;
    this->dynamicisWeighted = isWeighted;
    
    if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (dynamicgraphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));   //edgelist作为cudamallochost  锁页内存？
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
    }
    ///add
}