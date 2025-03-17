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


int main(int argc, char** argv)
{
	/*
	Test<int> test;
	cout << test.sum(20, 30) << endl;
	*/
	cudaFree(0);
	ArgumentParser arguments(argc, argv, true, false);//路径，是否weight
	
	Timer timer,timer1,timer2;   //Timer类
	timer.Start();
	timer1.Start();
	
	Graph<OutEdgeWeighted> graph(arguments.input, true);
	graph.ReadGraph();
	//graph.dynamicReadGraph();
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	//for(unsigned int i=0; i<100; i++)
	//	cout << graph.edgeList[i].end << " " << graph.edgeList[i].w8;

			//multi stream
     		cudaDeviceProp  prop;
			int whichDevice;
			cudaGetDevice( &whichDevice );
		    cudaGetDeviceProperties( &prop, whichDevice );
			if (!prop.deviceOverlap) {
				printf( "Device will not handle overlaps, so no speed up from streams\n" );
				return 0;
			}

			cudaEvent_t start, stop;
			float elapsedTime;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			cudaStream_t stream0,stream1,stream2,stream3,stream4,stream5,stream6,stream7;
			cudaStreamCreate(&stream0);
			cudaStreamCreate(&stream1); 
			cudaStreamCreate(&stream2);
			cudaStreamCreate(&stream3); 
			cudaStreamCreate(&stream4);
			cudaStreamCreate(&stream5); 
			cudaStreamCreate(&stream6);
			cudaStreamCreate(&stream7); 
			//end

			
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = DIST_INFINITY;
		graph.value1[i]=DIST_INFINITY;
		graph.value2[i] = DIST_INFINITY;
		graph.value3[i]=DIST_INFINITY;
		graph.value4[i] = DIST_INFINITY;
		graph.value5[i]=DIST_INFINITY;
		graph.value6[i] = DIST_INFINITY;
		graph.value7[i]=DIST_INFINITY;
		graph.label1[i] = true;
		graph.label2[i] = false;
	}
	graph.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;

	 //gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice)); //these four GPU design on graph.cu with cudamalloc
	 //gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	 //gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	 //gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	cudaMemcpyAsync(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream0);// newadd
	cudaMemcpyAsync(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream0);// newadd
	cudaMemcpyAsync(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream0);// newadd
	cudaMemcpyAsync(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream0);// newadd

	//cudaMemcpyAsync(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream1);// newadd
	cudaMemcpyAsync(graph.d_value1, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream1);// newadd
	cudaMemcpyAsync(graph.d_label11, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream1);// newadd
	cudaMemcpyAsync(graph.d_label22, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream1);// newadd

	cudaMemcpyAsync(graph.d_value2, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream2);// newadd
	cudaMemcpyAsync(graph.d_label111, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream2);// newadd
	cudaMemcpyAsync(graph.d_label222, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream2);// newadd

	cudaMemcpyAsync(graph.d_value3, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream3);// newadd
	cudaMemcpyAsync(graph.d_label1111, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream3);// newadd
	cudaMemcpyAsync(graph.d_label2222, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream3);// newadd

	cudaMemcpyAsync(graph.d_value4, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream4);// newadd
	cudaMemcpyAsync(graph.d_label5, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream4);// newadd
	cudaMemcpyAsync(graph.d_label6, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream4);// newadd

	cudaMemcpyAsync(graph.d_value5, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream5);// newadd
	cudaMemcpyAsync(graph.d_label55, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream5);// newadd
	cudaMemcpyAsync(graph.d_label66, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream5);// newadd

	cudaMemcpyAsync(graph.d_value6, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream6);// newadd
	cudaMemcpyAsync(graph.d_label555, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream6);// newadd
	cudaMemcpyAsync(graph.d_label666, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream6);// newadd

	cudaMemcpyAsync(graph.d_value7, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice,stream7);// newadd
	cudaMemcpyAsync(graph.d_label5555, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream7);// newadd
	cudaMemcpyAsync(graph.d_label6666, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice,stream7);// newadd


	//printf("1111\n");
	Subgraph<OutEdgeWeighted> subgraph(graph.num_nodes, graph.num_edges);//duixiang
	
	SubgraphGenerator<OutEdgeWeighted> subgen(graph); //duixinag
	//printf("*****************\n");
	subgen.generate(graph, subgraph);//对象.generate(图，子图)  -------> 生成graph的子图
	//printf("*****************\n");
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[i] = false;
	}
	graph.label1[arguments.sourceNode] = true;
	//gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));	

	gpuErrorcheck(cudaMemcpyAsync(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream0));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label11, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream1));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label111, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream2));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label1111, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream3));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label5, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream4));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label55, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream5));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label555, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream6));	
	gpuErrorcheck(cudaMemcpyAsync(graph.d_label5555, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice, stream7));

	cudaDeviceSynchronize();
	Partitioner<OutEdgeWeighted> partitioner;
	//printf("*****************##############\n");

	timer.Start();
	
	uint gItr = 0;
	
	bool finished,finished1,finished2,finished3,finished4,finished5,finished6,finished7;
	bool *d_finished;
	bool *d_finished1;
	bool *d_finished2;
	bool *d_finished3;
	bool *d_finished4;
	bool *d_finished5;
	bool *d_finished6;
	bool *d_finished7;
	clock_t start1;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished1, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished2, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished3, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished4, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished5, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished6, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished7, sizeof(bool)));
	//cudaMallocHost((void**)&finished,sizeof(bool));
	while (subgraph.numActiveNodes>0)    //整个子图
	{
		gItr++;//全局迭代次数
		//printf("test******************\n");
		partitioner.partition(subgraph, subgraph.numActiveNodes);// Subgraph分区
		// for(int i=0; i<partitioner.numPartitions; i++)
		// {
		// 	if(i==2||i==4||i==8)
		// 	partitioner.dynamicGraph(partitioner.partitionNodeSize[i],partitioner.fromNode[i],partitioner.fromEdge[i]);//dynamic分区的构造
		// 	//partitioner.dynamicpartition()//传第二分区
		// }
		cout<<"numPartitions=="<<partitioner.numPartitions<<endl;
		// a super iteration
		
		for(int i=0; i<partitioner.numPartitions; i++) //<
		{
			if(i==1||i==2||i==3||i==4||i==5)
			{
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream1));
				cudaDeviceSynchronize();
				mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream1>>>(subgraph.d_activeNodes, graph.d_label11, graph.d_label22, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
				cudaDeviceSynchronize();
				uint itr = 0;
				do
			{
				itr++;
				finished1=true;
				gpuErrorcheck(cudaMemcpyAsync(d_finished1, &finished1, sizeof(bool), cudaMemcpyHostToDevice,stream1));
				//gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1,256,0,stream1>>>(partitioner.partitionNodeSize[i],//host   partitioner.partitionNodeSize[i]/512 + 1 , 512
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value1, //cudamemcpy
						d_finished1,//wait
						(itr%2==1) ? graph.d_label11 : graph.d_label22,//cudamemcpy
						(itr%2==1) ? graph.d_label22 : graph.d_label11);		//cudamemcpy
				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				gpuErrorcheck(cudaMemcpy(&finished1, d_finished1, sizeof(bool), cudaMemcpyDeviceToHost));
				cudaDeviceSynchronize();
				cudaStreamSynchronize(stream1);
					//printf("^^^^^^^^^^^^^^^^^^^^\n");
					//cout<<"finished/finished1="<<finished<<""<<finished1<<endl;
			}while(!(finished1)); // finished=ture，finished=false才会循环 ,  ||finished1
			cudaDeviceSynchronize();
			cudaStreamSynchronize(stream1);
			// cudaStreamSynchronize(stream1);
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
			}

				if(i==2)
				{
					cudaDeviceSynchronize();
					gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream2));
					cudaDeviceSynchronize();
					mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream2>>>(subgraph.d_activeNodes, graph.d_label111, graph.d_label222, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
					cudaDeviceSynchronize();
					uint itr = 0;
					do
				{
					itr++;
					finished2 = true;
					gpuErrorcheck(cudaMemcpyAsync(d_finished2, &finished2, sizeof(bool), cudaMemcpyHostToDevice,stream2));
					//gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
					cudaDeviceSynchronize();
	
						sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1,256,0,stream2>>>(partitioner.partitionNodeSize[i],//host   partitioner.partitionNodeSize[i]/512 + 1 , 512
							partitioner.fromNode[i],//host
							partitioner.fromEdge[i],//host
							subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
							subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
							subgraph.d_activeEdgeList,  //cudamemcpy
							graph.d_outDegree,//cudamemcpy
							graph.d_value2, //cudamemcpy
							d_finished2,//wait
							(itr%2==1) ? graph.d_label111 : graph.d_label222,//cudamemcpy
							(itr%2==1) ? graph.d_label222 : graph.d_label111);		//cudamemcpy
					cudaDeviceSynchronize();
					gpuErrorcheck( cudaPeekAtLastError() );	
					gpuErrorcheck(cudaMemcpy(&finished2, d_finished2, sizeof(bool), cudaMemcpyDeviceToHost));
					cudaDeviceSynchronize();
					cudaStreamSynchronize(stream2);
						//printf("^^^^^^^^^^^^^^^^^^^^\n");
						//cout<<"finished/finished1="<<finished<<""<<finished1<<endl;
				}while(!(finished2)); // finished=ture，finished=false才会循环 ,  ||finished1
				cudaDeviceSynchronize();
				cudaStreamSynchronize(stream2);
				// cudaStreamSynchronize(stream1);
				cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
				}	


			if(i==3)
			{
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream1));
				cudaDeviceSynchronize();
				mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream3>>>(subgraph.d_activeNodes, graph.d_label1111, graph.d_label2222, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
				cudaDeviceSynchronize();
				uint itr = 0;
				do
			{
				itr++;
				finished1=true;
				gpuErrorcheck(cudaMemcpyAsync(d_finished3, &finished3, sizeof(bool), cudaMemcpyHostToDevice,stream1));
				//gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1,256,0,stream1>>>(partitioner.partitionNodeSize[i],//host   partitioner.partitionNodeSize[i]/512 + 1 , 512
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value3, //cudamemcpy
						d_finished3,//wait
						(itr%2==1) ? graph.d_label1111 : graph.d_label2222,//cudamemcpy
						(itr%2==1) ? graph.d_label2222 : graph.d_label1111);		//cudamemcpy
				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				gpuErrorcheck(cudaMemcpy(&finished3, d_finished3, sizeof(bool), cudaMemcpyDeviceToHost));
				cudaDeviceSynchronize();
				cudaStreamSynchronize(stream3);
					//printf("^^^^^^^^^^^^^^^^^^^^\n");
					//cout<<"finished/finished1="<<finished<<""<<finished1<<endl;
			}while(!(finished3)); // finished=ture，finished=false才会循环 ,  ||finished1
			cudaDeviceSynchronize();
			cudaStreamSynchronize(stream3);
			// cudaStreamSynchronize(stream1);
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
			}



		if(i==0)
		{			
			gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream0));
			//[gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream1));
			//******** CPU-GPU的data transfer
			//cudaDeviceSynchronize();
			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1,graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream0>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream1>>>(subgraph.d_activeNodes, graph.d_label11, graph.d_label22, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 , 256,0,stream2>>>(subgraph.d_activeNodes, graph.d_label111, graph.d_label222, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 , 256,0,stream3>>>(subgraph.d_activeNodes, graph.d_label1111, graph.d_label2222, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream4>>>(subgraph.d_activeNodes, graph.d_label5, graph.d_label6, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 ,256,0,stream5>>>(subgraph.d_activeNodes, graph.d_label55, graph.d_label66, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 , 256,0,stream6>>>(subgraph.d_activeNodes, graph.d_label555, graph.d_label666, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/256 + 1 , 256,0,stream7>>>(subgraph.d_activeNodes, graph.d_label5555, graph.d_label6666, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			cudaDeviceSynchronize();
			uint itr = 0;
			do
			{
				itr++;
				finished = true,finished1=true,finished2=true,finished3=true,finished4= true,finished5=true,finished6=true,finished7=true;;
				gpuErrorcheck(cudaMemcpyAsync(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice,stream0));
				gpuErrorcheck(cudaMemcpyAsync(d_finished1, &finished1, sizeof(bool), cudaMemcpyHostToDevice,stream1));
				gpuErrorcheck(cudaMemcpyAsync(d_finished2, &finished2, sizeof(bool), cudaMemcpyHostToDevice,stream2));
				gpuErrorcheck(cudaMemcpyAsync(d_finished3, &finished3, sizeof(bool), cudaMemcpyHostToDevice,stream3));
				gpuErrorcheck(cudaMemcpyAsync(d_finished4, &finished4, sizeof(bool), cudaMemcpyHostToDevice,stream4));
				gpuErrorcheck(cudaMemcpyAsync(d_finished5, &finished5, sizeof(bool), cudaMemcpyHostToDevice,stream5));
				gpuErrorcheck(cudaMemcpyAsync(d_finished6, &finished6, sizeof(bool), cudaMemcpyHostToDevice,stream6));
				gpuErrorcheck(cudaMemcpyAsync(d_finished7, &finished7, sizeof(bool), cudaMemcpyHostToDevice,stream7));
				//gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1,256,0,stream0>>>(partitioner.partitionNodeSize[i],//host   partitioner.partitionNodeSize[i]/512 + 1 , 512
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value, //cudamemcpy
						d_finished,//wait
						(itr%2==1) ? graph.d_label1 : graph.d_label2,//cudamemcpy
						(itr%2==1) ? graph.d_label2 : graph.d_label1);		//cudamemcpy
						
					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream1>>>(partitioner.partitionNodeSize[i],//host
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value1, //cudamemcpy
						d_finished1,//wait
						(itr%2==1) ? graph.d_label11 : graph.d_label22,//cudamemcpy
						(itr%2==1) ? graph.d_label22 : graph.d_label11);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream2>>>(partitioner.partitionNodeSize[i],//host
							partitioner.fromNode[i],//host
							partitioner.fromEdge[i],//host
							subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
							subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
							subgraph.d_activeEdgeList,  //cudamemcpy
							graph.d_outDegree,//cudamemcpy
							graph.d_value2, //cudamemcpy
							d_finished2,//wait
							(itr%2==1) ? graph.d_label111 : graph.d_label222,//cudamemcpy
							(itr%2==1) ? graph.d_label222 : graph.d_label111);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream3>>>(partitioner.partitionNodeSize[i],//host
								partitioner.fromNode[i],//host
								partitioner.fromEdge[i],//host
								subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
								subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
								subgraph.d_activeEdgeList,  //cudamemcpy
								graph.d_outDegree,//cudamemcpy
								graph.d_value3, //cudamemcpy
								d_finished3,//wait
								(itr%2==1) ? graph.d_label1111 : graph.d_label2222,//cudamemcpy
								(itr%2==1) ? graph.d_label2222 : graph.d_label1111);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream4>>>(partitioner.partitionNodeSize[i],//host
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value4, //cudamemcpy
						d_finished4,//wait
						(itr%2==1) ? graph.d_label5 : graph.d_label6,//cudamemcpy
						(itr%2==1) ? graph.d_label6 : graph.d_label5);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream5>>>(partitioner.partitionNodeSize[i],//host
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value5, //cudamemcpy
						d_finished5,//wait
						(itr%2==1) ? graph.d_label55 : graph.d_label66,//cudamemcpy
						(itr%2==1) ? graph.d_label66 : graph.d_label55);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream6>>>(partitioner.partitionNodeSize[i],//host
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value6, //cudamemcpy
						d_finished6,//wait
						(itr%2==1) ? graph.d_label555 : graph.d_label666,//cudamemcpy
						(itr%2==1) ? graph.d_label666 : graph.d_label555);//cudamemcpy

					sssp_async<<<partitioner.partitionNodeSize[i]/256 + 1, 256,0,stream7>>>(partitioner.partitionNodeSize[i],//host
						partitioner.fromNode[i],//host
						partitioner.fromEdge[i],//host
						subgraph.d_activeNodes,  //  不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeNodesPointer, //不需要memcpy赋值，只cudamalloc,通过核函数在GPU中赋值
						subgraph.d_activeEdgeList,  //cudamemcpy
						graph.d_outDegree,//cudamemcpy
						graph.d_value7, //cudamemcpy
						d_finished7,//wait
						(itr%2==1) ? graph.d_label5555 : graph.d_label6666,//cudamemcpy
						(itr%2==1) ? graph.d_label6666 : graph.d_label5555);//cudamemcpy
				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				// //cudaStreamSynchronize(stream0);
				// cudaStreamQuery(stream0);
				// //cudaStreamSynchronize(stream1);
				// cudaStreamQuery(stream1);
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished1, d_finished1, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished2, d_finished2, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished3, d_finished3, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished4, d_finished4, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished5, d_finished5, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished6, d_finished6, sizeof(bool), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(&finished7, d_finished7, sizeof(bool), cudaMemcpyDeviceToHost));
				cudaDeviceSynchronize();
				cudaStreamSynchronize(stream0);
				cudaStreamSynchronize(stream1);
				cudaStreamSynchronize(stream2);
				cudaStreamSynchronize(stream3);
				cudaStreamSynchronize(stream4);
				cudaStreamSynchronize(stream5);
				cudaStreamSynchronize(stream6);
				cudaStreamSynchronize(stream7);
					//printf("^^^^^^^^^^^^^^^^^^^^\n");
					//cout<<"finished/finished1="<<finished<<""<<finished1<<endl;
			}while(!(finished)||!(finished1)||!(finished2)||!(finished3)||!(finished4)||!(finished5)||!(finished6)||!(finished7)); // finished=ture，finished=false才会循环 ,  ||finished1
			cudaDeviceSynchronize();
			 cudaStreamSynchronize(stream0);
			// cudaStreamSynchronize(stream1);
			// cudaEventRecord(stop, 0);
			// cudaEventSynchronize(stop);
			// cudaEventElapsedTime(&elapsedTime, start, stop);
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
		}
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	cudaStreamSynchronize(stream4);
	cudaStreamSynchronize(stream5);
	cudaStreamSynchronize(stream6);
	cudaStreamSynchronize(stream7);
		subgen.generate(graph, subgraph);  //这里应该只生成一个子图	
	}
				//gpuErrorcheck(cudaMemcpyAsync(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice,stream0));

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";

	float readtime1 = timer1.Finish();
	cout << "Total finished in " << readtime1/1000 << " (s).\n";
	
	gpuErrorcheck(cudaMemcpyAsync(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream0));
	gpuErrorcheck(cudaMemcpyAsync(graph.value1, graph.d_value1, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream1));
	gpuErrorcheck(cudaMemcpyAsync(graph.value2, graph.d_value2, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream2));
	gpuErrorcheck(cudaMemcpyAsync(graph.value3, graph.d_value3, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream3));
	gpuErrorcheck(cudaMemcpyAsync(graph.value4, graph.d_value4, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream4));
	gpuErrorcheck(cudaMemcpyAsync(graph.value5, graph.d_value5, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream5));
	gpuErrorcheck(cudaMemcpyAsync(graph.value6, graph.d_value6, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream6));
	gpuErrorcheck(cudaMemcpyAsync(graph.value7, graph.d_value7, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream7));
	cudaDeviceSynchronize();
	//gpuErrorcheck(cudaMemcpyAsync(graph.value1, graph.d_value1, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost,stream1));
	printf("***************\n");
	// utilities::PrintResults(graph.value, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value1, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value2, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value3, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value4, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value5, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value6, min(30, graph.num_nodes));
	// utilities::PrintResults(graph.value7, min(30, graph.num_nodes));
	
	cudaDeviceSynchronize();

	// for(int i=0; i<20; i++)
	// 	cout << graph.value[i] << endl;
			
	 if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
	// 	//utilities::SaveResults(arguments.output, graph.value1, graph.num_nodes);
	// }

}

