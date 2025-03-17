
#include "partitioner.cuh"
#include "gpu_error_check.cuh"

template <class E>
Partitioner<E>::Partitioner()
{
	reset();
}


// template <class E>
// void Partitioner<E>::dynamicGraph(unsigned uint numNodes,unsigned int from, unsigned int numPartitionedEdges)
// {
// 	unsigned int sub2graphsize[4];
// 	for(i=0;i<3;i++)
// 	sub2graphsize[i]=numNodes/4;
// 	sub2graphsize[3]=numNodes-sub2graphsize[0]*3;
	

// }



template <class E>
void Partitioner<E>::partition(Subgraph<E> &subgraph, uint numActiveNodes)
{
	reset();
	
	unsigned int from, to;
	unsigned int left, right, mid;
	unsigned int partitionSize;//分区的尺寸
	unsigned int numNodesInPartition;//分区中顶点的数量
	unsigned int numPartitionedEdges;//分区的边数量
	bool foundTo;
	unsigned int accurCount;
	
	
	from = 0;
	to = numActiveNodes; // last in pointers
	numPartitionedEdges = 0;
	
	do
	{
		left = from;
		right = numActiveNodes;

		//cout << "#active nodes: " << numActiveNodes << endl;
		//cout << "left: " << left << "    right: " << right << endl;
		//cout << "pointer to left: " << subgraph.activeNodesPointer[left] << "    pointer to right: " << subgraph.activeNodesPointer[right] << endl;

		partitionSize = subgraph.activeNodesPointer[right] - subgraph.activeNodesPointer[left];
		//left=,right=
		//partitionSize = subgraph.activeNodesPointer[right] - subgraph.activeNodesPointer[left]/4;
		if(partitionSize <= subgraph.max_partition_size)
		{
			to = right;
		}
		else
		{
			foundTo = false;
			accurCount = 10;
			while(foundTo==false || accurCount>0)
			{
				mid = (left + right)/2;
				partitionSize = subgraph.activeNodesPointer[mid] - subgraph.activeNodesPointer[from];
				if(foundTo == true)
					accurCount--;
				if(partitionSize <= subgraph.max_partition_size)
				{
					left = mid;
					to = mid;
					foundTo = true;
				}
				else
				{
					right = mid;  
				}
			}
			

			if(to == numActiveNodes)
			{
				cout << "Error in Partitioning...\n";
				exit(-1);
			}

		}

		partitionSize = subgraph.activeNodesPointer[to] - subgraph.activeNodesPointer[from];
		numNodesInPartition = to - from;

		//cout << "from: " << from << "   to: " << to << endl;
		//cout << "#nodes in P: " << numNodesInPartition << "    #edges in P: " << partitionSize << endl;
		
		fromNode.push_back(from);//该分区顶点的开始索引
		fromEdge.push_back(numPartitionedEdges);//该分区边的开始索引
		partitionNodeSize.push_back(numNodesInPartition);//每个分区的顶点个数
		partitionEdgeSize.push_back(partitionSize);//每个分区的边尺寸
		
		from = to;
		numPartitionedEdges += partitionSize;
	
	} while (to != numActiveNodes);
	
	numPartitions = fromNode.size(); 
}

template <class E>
void Partitioner<E>::reset()
{
	fromNode.clear();
	fromEdge.clear();
	partitionNodeSize.clear();
	partitionEdgeSize.clear();
	numPartitions = 0;
}

template class Partitioner<OutEdge>;
template class Partitioner<OutEdgeWeighted>;
