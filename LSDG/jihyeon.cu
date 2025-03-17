#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/egraph_utilities.hpp"
#include "../shared/test.cuh"
#include "../shared/test.cu"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

int main()
{
	cudaFree(0);

	unsigned int sourceNode = 0;
	bool hasOutput;
	string output;

	std::string input1 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot1.el";
    std::string input2 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot2.el";
    std::string input3 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot3.el";
    std::string input4 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot4.el";

    
    std::string input_log1 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot1_changes.log";
    std::string input_log2 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot2_changes.log";
    std::string input_log3 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot3_changes.log";
    std::string input_log4 = "/home/netdb/jihyeon/make_snapshot/sk-2005_snapshot4_changes.log";



	Timer timer;
	timer.Start();
	
	Graph<OutEdgeWeighted> graph1(input1, false);
	// graph1.ReadGraph();
    Graph<OutEdgeWeighted> graph2(input2, false);
	// graph2.ReadGraph();
    Graph<OutEdgeWeighted> graph3(input3, false);
	// graph3.ReadGraph();
    Graph<OutEdgeWeighted> graph4(input4, false);
	// graph4.ReadGraph();
	
	// float readtime = timer.Finish();
	// cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	// for(unsigned int i=0; i<100; i++)
	// 	cout << graph1.edgeList[i].end << " " << graph1.edgeList[i].w8;
	
    //************* 연산 ************************
    
    
    
    //************* 연산 ************************
	std::vector<Graph<OutEdgeWeighted>> graphs = {graph1, graph2, graph3, graph4};
	float ProcessingResult;
	for(int i=0; i<4; i++)
	{	
		Graph<OutEdgeWeighted> graph = graphs[i];
		graph.ReadGraph();
		for(unsigned int i=0; i<graph.num_nodes; i++)
		{
			graph.value[i] = 0;
			graph.label1[i] = true;
			graph.label2[i] = false;
		}
		graph.value[sourceNode] = DIST_INFINITY;
		//graph.label[sourceNode] = true;


		gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
		
		Subgraph<OutEdgeWeighted> subgraph(graph.num_nodes, graph.num_edges);
		
		SubgraphGenerator<OutEdgeWeighted> subgen(graph);
		
		subgen.generate(graph, subgraph);
		
		for(unsigned int i=0; i<graph.num_nodes; i++)
		{
			graph.label1[i] = false;
		}
		graph.label1[sourceNode] = true;
		gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));	
		

		Partitioner<OutEdgeWeighted> partitioner;
		
		timer.Start();
		
		uint gItr = 0;
		
		bool finished;
		bool *d_finished;
		gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
			
		while (subgraph.numActiveNodes>0)
		{
			gItr++;
			
			partitioner.partition(subgraph, subgraph.numActiveNodes);
			// a super iteration
			for(int i=0; i<partitioner.numPartitions; i++)
			{
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();

				//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
				mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
				
				uint itr = 0;
				do
				{
					cout << "\t\tIteration " << ++itr << endl;
					finished = true;
					gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

					sswp_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
														partitioner.fromNode[i],
														partitioner.fromEdge[i],
														subgraph.d_activeNodes,
														subgraph.d_activeNodesPointer,
														subgraph.d_activeEdgeList,
														graph.d_outDegree,
														graph.d_value, 
														d_finished,
														(itr%2==1) ? graph.d_label1 : graph.d_label2,
														(itr%2==1) ? graph.d_label2 : graph.d_label1);	

					cudaDeviceSynchronize();
					gpuErrorcheck( cudaPeekAtLastError() );	
					
					gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
				}while(!(finished));
				
				cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;			
			}
			
			subgen.generate(graph, subgraph);
				
		}	
		
		float runtime = timer.Finish();
		cout << "Processing finished in " << runtime/1000 << " (s).\n";
		ProcessingResult += runtime;
		gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
		
		utilities::PrintResults(graph.value, min(30, graph.num_nodes));
				
		if(hasOutput)
			utilities::SaveResults(output, graph.value, graph.num_nodes);

		cudaFree(0);
		}
		cout << "All Processing finished in " << ProcessingResult/1000 << " (s).\n";
}
