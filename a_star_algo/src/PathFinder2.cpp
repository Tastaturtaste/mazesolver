/*cppimport
<%
setup_pybind11(cfg)
%>
*/

#include "PathFinder2.h"
#include "util\Timer.h"
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

#include "pybind11\pybind11.h"
#include "pybind11\stl.h"
#include "pybind11\numpy.h"
#include "PathFinder2.h"

namespace py = pybind11;

#define SET

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

template<class Container>
std::vector<int> set_to_vec(Container const & set) {
	std::vector<int> vec;
	vec.reserve(set.size());
	std::for_each(set.begin(), set.end(), [&vec](Node const* node) {vec.push_back(node->index); });
	return vec;
}

template<>
std::vector<int> set_to_vec(std::vector<int> const & set) {
	return set;
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
//constructing reverse path is more performant than normal construction
std::vector<int> construct_reverse_path(Node const* const start, Node const* const exit, size_t approx_path_length) {
	Timer timer("construction of reverse path");
	std::vector<int> path{ exit->index };
	path.reserve(approx_path_length);
	Node const* current = exit;
	Node const* parent = current->parent;
	do {
		path.push_back(parent->index);
		current = parent;
		parent = current->parent;
	} while (current != parent);
	return path;
}

std::tuple<std::vector<int>, std::vector<int>> get_path(int width, int height, std::vector<double> costs, int start_index, int exit_index, bool const diagonal_ok) {
	// find path from exit to start to make construction of path simpler
	std::swap(start_index, exit_index);
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
				node.closed = false;
				node.open = false;
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
		#ifdef SET
		bool operator()(Node const* const left, Node const* const right) const {
			if (left->combined_cost < right->combined_cost)
				return true;
			if (right->combined_cost < left->combined_cost)
				return false;
			if (left < right)
				return true;
			return false;
		}
		#else
		bool operator()(const Node* const left, const Node* const right) const { 
			if (left->combined_cost > right->combined_cost)
				return true;
			if (right->combined_cost > left->combined_cost)
				return false;
			if (left > right)
				return true;
			return false;
		}
		#endif	
	};
	#ifdef SET
	t_openlist<CompareCost> openlist(CompareCost{});
	#else
	Openlist<Node const*, CompareCost> openlist_new;
	openlist_new.reserve((width + height) / 2 * 5);
	#endif
	std::vector<int> closedlist;
	closedlist.reserve(width* height);
	
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
				* (openlist.begin()); openlist.erase(openlist.begin());
			#else
				openlist_new.pop();
			#endif
			if (current == exit_node) {
				std::cout << "Exit found! \n";
				timer.~Timer();
				return { { construct_reverse_path(start_node,exit_node,(width + height) * 2) }, set_to_vec<decltype(closedlist)>(closedlist) };
			}
			auto cur_pos = pos_from_index(current->index, width);
			int x{ cur_pos.first }, y{ cur_pos.second };
			
			closedlist.push_back(current->index);
			current->closed;
			for (int dx = -1; dx <= 1; ++dx) {
				for (int dy = -1; dy <= 1; ++dy) {
					if (!diagonal_ok && (std::abs(dx) == std::abs(dy)))
						continue;
					if (!is_valid(x + dx, y + dy, width, height, costs))
						continue;
					Node& neighbor = node_map[x + dx][y + dy];
					if ((&neighbor == current) || neighbor.closed)
						continue;
					double new_sure_cost = current->sure_cost + costs[neighbor.index];
					if (new_sure_cost <= neighbor.sure_cost || &neighbor == exit_node) {
						#ifdef SET
						if (neighbor.open) {
							openlist.erase(&neighbor);
							#else
							openlist_new.erase(&neighbor);	//Biggest Performance bottleneck with 62% time spend here on small_maze
							#endif
						} else {
							neighbor.open = true;
						}
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