#include "pybind11/pybind11.h"
#include "PathFinder2.h"
namespace py = pybind11;

PYBIND11_MODULE(pathfinder, m){
m.def("get_path", &get_path, "Get best path using an a*-algorithm", py::arg("height"), py::arg("width"), py::arg("weights"), py::arg("blocker_cutoff"), py::arg("start"), py::arg("exit"), py::arg("diagonal_ok"));

}