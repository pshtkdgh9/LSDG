#ifndef GRAPH_CUH
#define GRAPH_CUH


#include "globals.hpp"

template <class E>
class Graph
{
private:

public:
	string graphFilePath;
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
	uint *value1;
	uint *value2;
	uint *value3;
	uint *value4;
	uint *value5;
	uint *value6;
	uint *value7;
	uint *d_outDegree;
	uint *d_value;
	uint *d_value1;
	uint *d_value2;
	uint *d_value3;
	uint *d_value4;
	uint *d_value5;
	uint *d_value6;
	uint *d_value7;
	
	bool *d_label1;
	bool *d_label2;
	bool *d_label11;
	bool *d_label22;
	bool *d_label111;
	bool *d_label222;
	bool *d_label1111;
	bool *d_label2222;
	bool *d_label5;
	bool *d_label6;
	bool *d_label55;
	bool *d_label66;
	bool *d_label555;
	bool *d_label666;
	bool *d_label5555;
	bool *d_label6666;

	string graphFormat;
    Graph(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReadGraph();
};

#endif	//	GRAPH_CUH



