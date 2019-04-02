#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

std::vector<int> parse_file(std::string input, const char delimiter)
{
    std::ifstream infile(input);
    std::vector<int> output;
    std::string token;
    int i = 0;
    while(std::getline(infile, token, ','))
    {
		std::stringstream sstream;
        sstream << token;
        sstream >> i;
        output.push_back(i);
    }
    return output;
}

std::ostream& operator<<(std::ostream& ostream, const std::vector<int> vec)
{
    for(auto i = vec.begin(); i < vec.end(); ++i)
        ostream << *i << ',';

    return ostream;
}