#pragma once
#include <tuple>
#include <vector>
#include <unordered_set>

using index_t = long;

std::tuple<std::vector<index_t>, std::unordered_set<index_t>> get_path(int width, int height, std::vector<double> costs, index_t start_index, index_t exit_index, bool const diagonal_ok);
