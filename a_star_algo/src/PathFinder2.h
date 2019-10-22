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
	int index{};
	double sure_cost{};
	double heuristic_cost{};
	double combined_cost{};
	Node const* parent{ nullptr };
};

inline double combined_cost(const Node& node) noexcept { return node.sure_cost + node.heuristic_cost; }

struct CompareIndex {
	bool operator()(const Node& left, const Node& right) const noexcept { return left.index < right.index; }
};

template<typename PriorityComparator>
using t_openlist = std::set<Node const *, PriorityComparator>;
using t_closedlist = std::unordered_set<Node const *>;

std::tuple<std::vector<int>, std::vector<int>> get_path(int const width, int const height, std::vector<double> const costs, const int start_index, const int exit_index, bool const diagonal_ok);

//namespace old_slow_impl {
//	std::vector<int> get_path(const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, const int start, const int exit, bool diagonal_ok);
//	void expandNode(const Node& currentNode, const int exit, t_openlist& openlist, t_closedlist& closedlist, const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, std::vector<int>& costs, std::unordered_map<int, int>& connections, const bool diagonal_ok);
//}

std::vector<int> parse_string_to_weights(std::string s);
