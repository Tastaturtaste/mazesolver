/*cppimport
<%
setup_pybind11(cfg)
%>
*/

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "PathFinder2.h"

namespace py = pybind11;


PYBIND11_MODULE(pathfinder, m)
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