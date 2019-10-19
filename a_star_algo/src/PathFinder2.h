#pragma once
//https://www.codewars.com/kata/57658bfa28ed87ecfa00058a/train/cpp

#include <vector>
#include <queue>
#include <set>
#include <string>
#include <iostream>
#include <unordered_map>

std::ostream& operator<<(std::ostream& ostream, const std::pair<int, int>& pair);

class Node {
public:
	int m_index; // {y,x}
	double m_sure_cost;
	double m_heuristic_cost;
	Node(int index, double sure_cost);
	inline double get_combined_cost() const { return m_sure_cost + m_heuristic_cost; }
};

struct CompareIndex {
	bool operator()(const Node& left, const Node& right) const { return left.m_index < right.m_index; }
};

// < is faster, don't know why
struct ComparePriority {
	bool operator()(const Node& left, const Node& right) const { return left.get_combined_cost() > right.get_combined_cost(); }
};

using t_openlist = std::priority_queue<Node, std::vector<Node>, ComparePriority>;
using t_closedlist = std::set<Node, CompareIndex>;

double heuristic_cost(const int node1_y, const int node2_y, const int node1_x, const int node2_x, bool diagonal_ok = false);


const bool closed_contains(const std::set<Node>& set, const Node& element);

void expandNode(const Node& currentNode, const int exit, t_openlist& openlist, t_closedlist& closedlist, const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, std::vector<int>& costs, std::unordered_map<int, int>& connections, const bool diagonal_ok);

std::vector<int> get_path(const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, const int start, const int exit, bool diagonal_ok);

std::vector<int> parse_string_to_weights(std::string s);
