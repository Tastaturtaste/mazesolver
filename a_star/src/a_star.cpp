/*cppimport
<%
setup_pybind11(cfg)
%>
*/

#include "a_star.h"
#include "util\Timer.h"

#include <ctype.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <exception>
#include <algorithm>

#include "pybind11\pybind11.h"
#include "pybind11\stl.h"
#include "pybind11\numpy.h"

namespace py = pybind11;


struct Node {
	long index{};
	double sure_cost{};
	double heuristic_cost{};
	double combined_cost{};
	Node const* parent{ nullptr };

	friend bool operator<(Node const& l, Node const& r) {
		if (l.combined_cost < r.combined_cost)
			return true;
		else if (l.index < r.index)
			return true;
		else
			return false;
	}
};

inline double combined_cost(const Node& node) noexcept { return node.sure_cost + node.heuristic_cost; }


std::ostream& operator<<(std::ostream& ostream, const std::pair<int, int>& pair)
{
	ostream << '{' << pair.first << ", " << pair.second << '}';
	return ostream;
}

template<typename T>
double heuristic_cost(const std::pair<T,T> node_1, const std::pair<T,T> node_2, bool diagonal_ok)
{
	const double dist1 = std::abs(node_1.first - node_2.first);
	const double dist2 = std::abs(node_1.second - node_2.second);
	if (diagonal_ok) {
		return std::sqrt(dist1*dist1 + dist2*dist2);
	}
	else {
		return dist1 + dist2;
	}
}

std::pair<int, int> pos_from_index(int index, int width) {
	return { index % width, index / width };
}

template<class Container >
std::unordered_set<index_t> vec_to_set(Container const & vec) {
	std::unordered_set<index_t> set;
	set.reserve(vec.size() / 4);
	for (size_t i = 0; i < vec.size(); ++i) {
		if (vec[i]) {
			set.insert(i);
		}
	}
	return set;
}

std::vector<index_t> construct_path(Node const* const start, Node const* const exit, size_t approx_path_length = 1) {
	// Beginning with the start Node traverses through the Node pointers and adds the indices to the path
	// Expects every Node to point to the following Node in the path
	std::vector<index_t> path{ start->index };
	path.reserve(approx_path_length);
	Node const* current = start;
	Node const* parent = current->parent;
	do {
		path.push_back(parent->index);
		current = parent;
		parent = current->parent;
	} while (current != exit);
	return path;
}

std::tuple<std::vector<index_t>, std::unordered_set<index_t>> get_path(int width, int height, std::vector<double> costs, index_t start_index, index_t exit_index, bool const diagonal_ok) {
	if (width < 0 || height < 0) {
		throw std::domain_error("width and height have to be positive!");
	}
	if (width * height != costs.size()) {
		throw std::length_error("width * height != len(costs)!");
	}
	if (start_index < 0 || start_index >(costs.size() - 1) || exit_index < 0 || exit_index >(costs.size() - 1)) {
		throw std::out_of_range("start and end indices have to be in the range [0, len(costs) )!");
	}

	// find path from exit to start, this way when traversing the nodes from the start
	// every node points to the next one in the path
	std::swap(start_index, exit_index);
	std::pair<int,int> start_pos{ start_index % width, start_index / width };
	std::pair<int, int> exit_pos{ exit_index % width, exit_index / width };
	std::vector<Node> node_map = [&]() {
		std::vector<Node> nodes;
		nodes.reserve(costs.size());
		for (auto idx = 0l; idx < width*height; ++idx) {
			nodes.push_back({ idx, std::numeric_limits<double>::infinity(), 0.0, std::numeric_limits<double>::infinity(), nullptr });
		}
		return nodes;
	}();

	Node* const exit_node = &(node_map[exit_index]);
	Node const* const start_node = &(node_map[start_index]);

	node_map[start_index].sure_cost = 0.0;
	node_map[start_index].heuristic_cost = heuristic_cost(start_pos, exit_pos,diagonal_ok);
	node_map[start_index].combined_cost = start_node->sure_cost + start_node->heuristic_cost;

	constexpr double diag_cost_mod = 1.414213562; // sqrt(2)
	
	auto cmp = [](Node const* l, Node const* r) { return *l < *r; };
	std::set<Node const*, decltype(cmp)> openlist(std::move(cmp));
	std::unordered_set<index_t> closedlist{};
	closedlist.reserve(costs.size() / 2);
	openlist.insert( &(node_map[start_index]) );
	while (	!openlist.empty() ){
		Node const* const current = *(openlist.begin());
		if (current == exit_node) {
			// call with exit and start switched to get correct direction back
			return { construct_path(exit_node, start_node, width*height/4) , closedlist };
		}
		openlist.erase(openlist.begin());
		closedlist.insert(current->index);
		auto [cur_x, cur_y] = pos_from_index(current->index, width);
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				// skip diagonal entrys if diagonals are not viable
				if (!diagonal_ok && (std::abs(dx) == std::abs(dy)))
					continue;
				// skip if node would go outside rectangle
				auto x = cur_x + dx; auto y = cur_y + dy;
				if (static_cast<uint32_t>(x) >= width || static_cast<uint32_t>(y) >= height ) // Project negative values to big positives so only one comparison per value needed
					continue;
				Node& neighbor = node_map.at(current->index + dx + dy*width);
				// skip previously visited nodes, including the current node
				if (closedlist.find(neighbor.index) != closedlist.end())
					continue;
				// skip if node is not passable
				if (costs[neighbor.index] < 0.0)
					continue;
				const bool diagonal_move = dx * dy != 0; // should be inlined
				double new_sure_cost = current->sure_cost + (diagonal_move ? diag_cost_mod : 1.0) * costs[neighbor.index];
				if (new_sure_cost < neighbor.sure_cost) {
					// Make sure to not invalidate the ordered set
					if (auto it = openlist.find(&neighbor); it != openlist.end())
						openlist.erase(it);
					neighbor.sure_cost = new_sure_cost;
					neighbor.heuristic_cost = heuristic_cost({x,y}, exit_pos, diagonal_ok);
					// combined cost for ordering of the open set
					neighbor.combined_cost = neighbor.sure_cost + neighbor.heuristic_cost;
					neighbor.parent = current;
					openlist.insert(&neighbor);
				}
			}
		}
	}
	return { {-1}, closedlist };
}

PYBIND11_MODULE(a_star, m)
{
	m.doc() = R"docstring(
			This module exposes only one function named get_path.)docstring";
	m.def("get_path", &get_path, R"docstring(
			Searches for the best path from start to finish
			using an a*-algorithm.
			The function expects a flattened indexvector starting from top left
			and row-major ordering.
			Unpassable Nodes are assumed to have negative values.
			Return value is a list of the indices of the found path and all explored pixels.
			)docstring",
		py::arg("width"), py::arg("height"), py::arg("costs"), py::arg("start_index"), py::arg("exit_index"), py::arg("diagonal_ok") = true
	);
}