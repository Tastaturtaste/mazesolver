%module pathfinder
%{
    #include "PathFinder2_swig.h"
%}
%include "std_vector.i";
%include "std_string.i";
namespace std {
    %template(IntVec) vector<int>;
}

std::vector<int> parse_string_to_weights(std::string s);
std::vector<int> get_path(const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, const int start, const int exit, bool diagonal_ok);