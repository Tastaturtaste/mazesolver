#pragma once
//https://www.codewars.com/kata/57658bfa28ed87ecfa00058a/train/cpp

#include <vector>
#include <queue>
#include <set>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

std::ostream& operator<<(std::ostream& ostream, const std::pair<int, int>& pair);

struct Node {
	int index;
	bool open;
	bool closed;
	double sure_cost;
	double heuristic_cost;
	double combined_cost;
	Node const* parent;
};

inline double combined_cost(const Node& node) noexcept { return node.sure_cost + node.heuristic_cost; }

struct CompareIndex {
	bool operator()(const Node& left, const Node& right) const noexcept { return left.index < right.index; }
};

template<typename PriorityComparator>
using t_openlist = std::set<Node const *, PriorityComparator>;

std::tuple<std::vector<int>, std::vector<int>> get_path(int width, int height, std::vector<double> costs, int start_index, int exit_index, bool const diagonal_ok);

std::vector<int> parse_string_to_weights(std::string s);
