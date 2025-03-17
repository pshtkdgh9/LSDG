#ifndef ARGUMENT_PARSING_HPP
#define ARGUMENT_PARSING_HPP

#include "globals.hpp"


class ArgumentParser  //参数解析
{
private:

public:
	int argc;
	char** argv;
	
	bool canHaveSource;
	bool canHaveItrs;
	
	bool hasInput;
	bool hasSourceNode;
	bool hasOutput;
	bool hasDeviceID;
	bool hasNumberOfItrs;
	string input;
	std::vector<std::string> inputs;
	int sourceNode;
	string output;
	int deviceID;
	int numberOfItrs;
	
	
	ArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs); //构造函数
	
	bool Parse();
	
	string GenerateHelpString();
	
};


#endif	//	ARGUMENT_PARSING_HPP
