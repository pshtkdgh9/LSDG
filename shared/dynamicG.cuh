#ifndef DYNAMICG_CUH
#define DYNAMICG_CUH


#include "globals.hpp"

template <class E>
class dynamicGraph
{
private:

public:
	string dynamicgraphFilePath;
	bool isWeighted;
	bool isLarge;
	uint num_nodes;
	uint num_edges;
	uint *nodePointer;
	E *edgeList;
	uint *outDegree;
	bool *label1;
	bool *label2;
	uint *value;
	uint *d_outDegree;
	uint *d_value;
	uint *d_value1;

	string graphFormat;
    dynamicGraph(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReaddynamicGraph();
};

#endif	//	GRAPH_CUH