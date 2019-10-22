/*cppimport
<%
setup_pybind11(cfg)
%>
*/

#include "PathFinder2.h"
#include "..\util\Timer.h"
#include "Openlist.h"

#include <vector>
#include <queue>
#include <set>
#include <cmath>
#include <string>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <algorithm>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "PathFinder2.h"

#define OPENLIST

std::ostream& operator<<(std::ostream& ostream, const std::pair<int, int>& pair)
{
	ostream << '{' << pair.first << ", " << pair.second << '}';
	return ostream;
}

template<typename T>
double heuristic_cost(const std::pair<T,T> node_1, const std::pair<T,T> node_2, bool diagonal_ok)
{
	if (diagonal_ok)
		return sqrt(std::pow(node_1.first - node_2.first, 2) + std::pow(node_1.second - node_2.second, 2));
	else
		return std::abs(node_1.first - node_2.first) + std::abs(node_1.second - node_2.second);
}

std::pair<int, int> pos_from_index(int index, int width) {
	return { index % width, index / width };
}

inline bool is_valid(int x, int y, int width, int height, std::vector<double> const& costs) {
	return x >= 0 && x < width && y >= 0 && y < height && costs[x + y*width] >= 0.0;
}

//bool closed_contains(const std::set<Node, CompareIndex>& set, const Node& element)
//{
//	return set.find(element) != set.end();
//}

template<class Container>
std::vector<int> set_to_vec(Container set) {
	std::vector<int> vec;
	vec.reserve(set.size());
	std::for_each(set.begin(), set.end(), [&vec](Node const* node) {vec.push_back(node->index); });
	return vec;
}

std::vector<int> construct_path(Node const* const start, Node const* const exit, size_t approx_path_length){
	Timer timer("construction of path");
	std::vector<int> path{exit->index};
	path.reserve(approx_path_length);
	Node const * current = exit;
	Node const* parent = current->parent;
	do {
		path.insert(path.begin(), parent->index);
		current = parent;
		parent = current->parent;
	} while (current != parent);
	return path;
}

std::tuple<std::vector<int>, std::vector<int>> get_path(int const width, int const height, std::vector<double> const costs, const int start_index, const int exit_index, bool const diagonal_ok) {
	std::pair<int,int> start_pos{ start_index % width, start_index / width };
	std::pair<int, int> exit_pos{ exit_index % width, exit_index / width };

	std::vector<std::vector<Node>> node_map(width,std::vector<Node>(height));
	{
		int x{};
		int y{};
		std::for_each(node_map.begin(), node_map.end(), [&x, &y, &width, &costs](std::vector<Node>& columns) {
			std::for_each(columns.begin(), columns.end(), [&x, &y, &width, &costs](Node& node) {
				node.index = x + y * width;
				node.sure_cost = std::numeric_limits<double>::infinity();
				++y;
			});
			++x;
			y = 0;
		});
	}

	Node* const exit_node = &(node_map[exit_pos.first][exit_pos.second]);
	Node const* const start_node = &(node_map[start_pos.first][start_pos.second]);

	node_map[start_pos.first][start_pos.second].sure_cost = 0.0;
	node_map[start_pos.first][start_pos.second].heuristic_cost = heuristic_cost(start_pos, exit_pos, diagonal_ok);
	node_map[start_pos.first][start_pos.second].combined_cost = start_node->sure_cost + start_node->heuristic_cost;
	node_map[start_pos.first][start_pos.second].parent = start_node;

	struct CompareHeuristic {
		bool operator()(const Node * const left, const Node * const right) const { return left->heuristic_cost > right->heuristic_cost; }
	};
	struct CompareCost {
		bool operator()(const Node* const left, const Node* const right) const { return left->combined_cost > right->combined_cost; }
	};
	#ifdef SET
	t_openlist<CompareCost> openlist(CompareCost{});
	#else
	Openlist<Node const*, CompareCost> openlist_new;
	openlist_new.reserve((width + height) / 2 * 5);
	#endif
	t_closedlist closedlist;
	
	#ifdef SET
	openlist.insert(&(node_map[start_pos.first][start_pos.second]));
	#else
	openlist_new.push(&(node_map[start_pos.first][start_pos.second]));
	#endif
	{
		Timer timer("Hot loop");
		while (	
			#ifdef SET 
			!openlist.empty()
			#else 
			!openlist_new.empty()
			#endif
			){
			Node const* const current = 
			#ifdef SET
				* (openlist.begin()); openlist.erase(current);
			#else
				openlist_new.pop();
			#endif
			if (current == exit_node) {
				std::cout << "Exit found! \n";
				return { { construct_path(start_node,exit_node,(width + height) * 2) }, set_to_vec<decltype(closedlist)>(closedlist) };
			}
			auto cur_pos = pos_from_index(current->index, width);
			int x{ cur_pos.first }, y{ cur_pos.second };
			closedlist.insert(current);
			for (int dx = -1; dx <= 1; ++dx) {
				for (int dy = -1; dy <= 1; ++dy) {
					if(!diagonal_ok && (std::abs(dx) == std::abs(dy)))
						continue;
					if (!is_valid(x+dx,y+dy,width,height,costs))
						continue;
					Node& neighbor = node_map[x + dx][y + dy];
					if ((&neighbor == current) || (closedlist.find(&neighbor) != closedlist.end()))
						continue;
					double new_sure_cost = current->sure_cost + costs[neighbor.index];
					if (new_sure_cost <= neighbor.sure_cost || &neighbor == exit_node){
						#ifdef SET
						openlist.erase(&neighbor);
						#else
						openlist_new.erase(&neighbor);	//Biggest Performance bottleneck with 62% time spend here on small_maze
						#endif
						neighbor.sure_cost = new_sure_cost;
						neighbor.heuristic_cost = heuristic_cost(pos_from_index(neighbor.index, width), exit_pos, diagonal_ok);
						neighbor.combined_cost = neighbor.sure_cost + neighbor.heuristic_cost;
						neighbor.parent = current;
						#ifdef SET
						openlist.insert(&neighbor);
						#else
						openlist_new.push(&neighbor);
						#endif
					}
				}
			}
		}
	}
	return { {-1}, set_to_vec(closedlist) };
}

//namespace old_slow_impl {
//	void expandNode(const Node& currentNode, const int exit, t_openlist& openlist, t_closedlist& closedlist, const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, std::vector<double>& costs, std::unordered_map<int, int>& connections, const bool diagonal_ok)
//	{
//		std::vector<Node> neighbors;
//
//		int row = currentNode.m_index / width;
//		int col = currentNode.m_index % width;
//		// add neighbors clockwise starting at top left
//		if (diagonal_ok && row > 0 && col > 0) neighbors.emplace_back(currentNode.m_index - (width + 1), currentNode.m_sure_cost);
//		if (row > 0) neighbors.emplace_back(currentNode.m_index - width, currentNode.m_sure_cost);
//		if (diagonal_ok && row > 0 && col < width - 1) neighbors.emplace_back(currentNode.m_index - width + 1, currentNode.m_sure_cost);
//		if (col < width - 1) neighbors.emplace_back(currentNode.m_index + 1, currentNode.m_sure_cost);
//		if (diagonal_ok && row > height - 1 && col > width - 1) neighbors.emplace_back(currentNode.m_index + width + 1, currentNode.m_sure_cost);
//		if (row < height - 1) neighbors.emplace_back(currentNode.m_index + width, currentNode.m_sure_cost);
//		if (diagonal_ok && row > height - 1 && col > 0) neighbors.emplace_back(currentNode.m_index + (width - 1), currentNode.m_sure_cost);
//		if (col > 0) neighbors.emplace_back(currentNode.m_index - 1, currentNode.m_sure_cost);
//		for (auto& neighbor : neighbors)
//		{
//			if (weights[neighbor.m_index] >= blocker_cutoff)
//			{
//				closedlist.insert(neighbor);
//				continue;
//			}
//			if (closed_contains(closedlist, neighbor))
//			{
//				continue;
//			}
//
//			neighbor.m_sure_cost += weights[neighbor.m_index];
//
//			if (costs[neighbor.m_index] <= neighbor.m_sure_cost)
//			{
//				continue;
//			}
//			costs[neighbor.m_index] = neighbor.m_sure_cost;
//			neighbor.m_heuristic_cost = heuristic_cost(neighbor.m_index / width, exit / width, neighbor.m_index % width, exit % width, diagonal_ok);
//			openlist.push(neighbor);
//			connections[neighbor.m_index] = currentNode.m_index;
//		}
//	}
//
//	std::vector<int> get_path(const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, const int start, const int exit, bool diagonal_ok)
//	{
//		std::cout << "Calculating path...\n";
//		t_openlist openlist;
//		t_closedlist closedlist;
//		bool found_path = false;
//		Node start_node(start, 0);
//		Node end_node(exit, 0);
//		std::unordered_map<int, int> connections; // connections[node_index] == previous_node_index
//		std::vector<double> costs(width * height, std::numeric_limits<double>::max());
//
//		openlist.push(start_node);
//		{
//			Timer timer("get_path hot loop");
//			while (openlist.size() > 0)
//			{
//				Node currentNode = openlist.top();
//				openlist.pop();
//				if (currentNode.m_index == exit)
//				{
//					found_path = true;
//					break;
//				}
//
//				closedlist.insert(currentNode);
//				expandNode(currentNode, exit, openlist, closedlist, height, width, weights, blocker_cutoff, costs, connections, diagonal_ok);
//			}
//		}
//		if (!found_path)
//			return {};
//
//		std::vector<int> path{ exit };
//		path.reserve(connections.size() / 2);
//		while (connections.find(path[0]) != connections.end())
//		{
//			path.insert(path.begin(), connections[path[0]]);
//		}
//		return path;
//	}
//}

std::vector<int> parse_string_to_weights(std::string s)
{
	s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
	std::vector<int> weights(s.size(), std::numeric_limits<int>::max());

	int row(0);
	int col(0);
	for (auto i = 0; i < s.size(); ++i)
	{
		if (s[i] == '.') weights[i] = 1;
	}
	return weights;
}

namespace py = pybind11;

PYBIND11_MODULE(a_star_algo, m)
{
	m.doc() = R"docstring(
			This module exposes only one function named get_path.)docstring";
	m.def("get_path", &get_path, R"docstring(
			Searches for the best path from start to finish
			using an a*-algorithm.
			The function expects a flattened indexvector starting from top left
			and row-major ordering.
			Unpassable Nodes are assumed to have negative values.
			Returnvalue is a list of either the indices of the found path or all tried pixels.
			)docstring",
		py::arg("width"), py::arg("height"), py::arg("costs"), py::arg("start_index"), py::arg("exit_index"), py::arg("diagonal_ok")
	);
}