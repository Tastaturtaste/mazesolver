/*cppimport
<%
setup_pybind11(cfg)
%>
*/

#include "PathFinder2.h"
#include "..\util\Timer.h"

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


std::ostream& operator<<(std::ostream& ostream, const std::pair<int, int>& pair)
{
	ostream << '{' << pair.first << ", " << pair.second << '}';
	return ostream;
}

Node::Node(int index, double sure_cost)
		: m_index(index), m_sure_cost(sure_cost) {}

double heuristic_cost(const int node1_y, const int node2_y, const int node1_x, const int node2_x, bool diagonal_ok)
{
	return diagonal_ok ? sqrt(std::pow(node1_x - node2_x, 2) + std::pow(node1_y - node2_y, 2)) : std::abs(node1_x - node2_x) + std::abs(node1_y - node2_y);
}

const bool closed_contains(const std::set<Node, CompareIndex>& set, const Node& element)
{
	return set.find(element) != set.end();
}

void expandNode(const Node& currentNode, const int exit, t_openlist& openlist, t_closedlist& closedlist, const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, std::vector<double>& costs, std::unordered_map<int, int>& connections, const bool diagonal_ok)
{
	std::vector<Node> neighbors;

	int row = currentNode.m_index / width;
	int col = currentNode.m_index % width;
	// add neighbors clockwise starting at top left
	if (diagonal_ok && row > 0 && col > 0) neighbors.emplace_back(currentNode.m_index - (width + 1), currentNode.m_sure_cost);
	if (row > 0) neighbors.emplace_back(currentNode.m_index - width, currentNode.m_sure_cost);
	if (diagonal_ok && row > 0 && col < width - 1) neighbors.emplace_back(currentNode.m_index - width + 1, currentNode.m_sure_cost);
	if (col < width - 1) neighbors.emplace_back(currentNode.m_index + 1, currentNode.m_sure_cost);
	if (diagonal_ok && row > height - 1 && col > width - 1) neighbors.emplace_back(currentNode.m_index + width + 1, currentNode.m_sure_cost);
	if (row < height - 1) neighbors.emplace_back(currentNode.m_index + width, currentNode.m_sure_cost);
	if (diagonal_ok && row > height - 1 && col > 0) neighbors.emplace_back(currentNode.m_index + (width - 1), currentNode.m_sure_cost);
	if (col > 0) neighbors.emplace_back(currentNode.m_index - 1, currentNode.m_sure_cost);
	for (auto& neighbor : neighbors)
	{
		if (weights[neighbor.m_index] >= blocker_cutoff)
		{
			closedlist.insert(neighbor);
			continue;
		}
		if (closed_contains(closedlist, neighbor))
		{
			continue;
		}

		neighbor.m_sure_cost += weights[neighbor.m_index];

		if (costs[neighbor.m_index] <= neighbor.m_sure_cost)
		{
			continue;
		}
		costs[neighbor.m_index] = neighbor.m_sure_cost;
		neighbor.m_heuristic_cost = heuristic_cost(neighbor.m_index / width, exit / width, neighbor.m_index % width, exit % width, diagonal_ok);
		openlist.push(neighbor);
		connections[neighbor.m_index] = currentNode.m_index;
	}
}

std::vector<int> get_path(const int height, const int width, std::vector<int>& weights, const int blocker_cutoff, const int start, const int exit, bool diagonal_ok)
{
	std::cout << "Calculating path...\n";
	t_openlist openlist;
	t_closedlist closedlist;
	bool found_path = false;
	Node start_node(start, 0);
	Node end_node(exit, 0);
	std::unordered_map<int, int> connections; // connections[node_index] == previous_node_index
	std::vector<double> costs(width*height, std::numeric_limits<double>::max());

	openlist.push(start_node);
	{
		Timer timer("get_path hot loop");
		while (openlist.size() > 0)
		{
			Node currentNode = openlist.top();
			openlist.pop();
			if (currentNode.m_index == exit)
			{
				found_path = true;
				break;
			}

			closedlist.insert(currentNode);
			expandNode(currentNode, exit, openlist, closedlist, height, width, weights, blocker_cutoff, costs, connections, diagonal_ok);
		}
	}
	if (!found_path)
		return {};

	std::vector<int> path{ exit };
	path.reserve(connections.size() / 2);
	while (connections.find(path[0]) != connections.end())
	{
		path.insert(path.begin(), connections[path[0]]);
	}
	return path;
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

namespace py = pybind11;


PYBIND11_MODULE(a_star_algo, m)
{
	m.doc() = R"docstring(
			This module exposes only one function named get_path.)docstring";
	m.def("get_path", &get_path, R"docstring(
			Searches for the best path from start to finish
			using an a*-algorithm.)docstring",
		py::arg("height"), py::arg("width"), py::arg("weights"), py::arg("blocker_cutoff"),
		py::arg("start"), py::arg("exit"), py::arg("diagonal_ok")
	);
}