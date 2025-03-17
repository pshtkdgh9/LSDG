#include "graph.cuh"
#include "gpu_error_check.cuh"

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}


template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
	int sourceNode; //수정
    int destinationNode; //수정
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		vector<Edge> edges;
		Edge newEdge;
		while(getline( infile, line ))
		{
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> newEdge.source;
			ss >> newEdge.end;
			
			edges.push_back(newEdge);
			edgeCounter++;
			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;				
		}
		infile.close();
		num_nodes = max + 1;
		num_edges = edgeCounter;
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));// 边文件分配锁页内存
		uint *degree = new uint[num_nodes];
		for(uint i=0; i<num_nodes; i++)
			degree[i] = 0;
		for(uint i=0; i<num_edges; i++)
			degree[edges[i].source]++;
		
		uint counter=0;
		for(uint i=0; i<num_nodes; i++)
		{
			nodePointer[i] = counter;
			counter = counter + degree[i];
		}
		nodePointer[num_nodes] = num_edges;
		uint *outDegreeCounter  = new uint[num_nodes];
		uint location;  
		for(uint i=0; i<num_edges; i++)
		{
			location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			//if(isWeighted)
			//	edgeList[location].w8 = edges[i].w8;
			outDegreeCounter[edges[i].source]++;  
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;						
		
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	//outDegree  = new unsigned int[num_nodes];   //分配内存
	cudaHostAlloc((void**)&outDegree,num_nodes*sizeof(unsigned int),cudaHostAllocDefault);    //newadd
	printf("2222222\n");
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	
	 //label1 = new bool[num_nodes];
	 //label2 = new bool[num_nodes];
	cudaHostAlloc((void**)&label1,num_nodes*sizeof(bool),cudaHostAllocDefault);  //newadd
	cudaHostAlloc((void**)&label2,num_nodes*sizeof(bool),cudaHostAllocDefault);  // newadd
	//value  = new unsigned int[num_nodes];
	cudaHostAlloc((void**)&value,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value1,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value2,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value3,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value4,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value5,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value6,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd
	cudaHostAlloc((void**)&value7,num_nodes*sizeof(unsigned int),cudaHostAllocDefault); //newadd

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value1, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value2, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value3, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value4, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value5, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value6, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value7, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label11, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label22, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label111, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label222, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1111, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2222, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label5, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label6, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label55, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label66, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label555, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label666, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label5555, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label6666, num_nodes * sizeof(bool)));
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
	
	/*
	for(int i=0; i<10; i++)
		cout << nodePointer[i] << " ";
	cout << "\n\n\n\n\n";
	
	cout <<  nodePointer[1] - nodePointer[0] << endl;
	for(int i=nodePointer[0]; i<nodePointer[1]; i++)
		cout << edgeList[i].end << " ";
	cout << "\n\n\n\n\n";
	
	cout <<  nodePointer[100001] - nodePointer[100000] << endl;
	for(int i=nodePointer[100000]; i<nodePointer[100001]; i++)
		cout << edgeList[i].end << " ";
	cout << "\n\n\n\n\n";
	
	cout <<  nodePointer[1000001] - nodePointer[1000000] << endl;
	for(int i=nodePointer[1000000]; i<nodePointer[1000001]; i++)
		cout << edgeList[i].end << " ";	
	cout << "\n\n\n\n\n";
	
	cout <<  nodePointer[num_nodes] - nodePointer[num_nodes-1] << endl;
	for(int i=nodePointer[num_nodes-1]; i<nodePointer[num_nodes]; i++)
		cout << edgeList[i].end << " ";
	cout << "\n\n\n\n\n";
	*/

}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;
