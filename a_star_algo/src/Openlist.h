#pragma once

#include <vector>
#include <algorithm>


template<class T, class Compare>
class Openlist {
	std::vector<T> container{};
	const Compare comp{};
public:
	Openlist() {
		std::make_heap(container.begin(), container.end(), comp);
	}
	void push(T const& val) {
		container.push_back(val);
		std::push_heap(container.begin(), container.end());
	}
	T pop() {
		std::pop_heap(container.begin(), container.end());
		T val = container.back();
		container.pop_back();
		return val;
	}
	inline size_t size() const { return container.size(); }
	inline size_t capacity() const { return container.capacity(); }
	void reserve(size_t new_cap) { container.reserve(new_cap); }
	bool contains(const T& value) {
		return std::find(container.begin(), container.end(), value) != container.end();
	}
	inline bool empty() const { return container.empty(); }
	bool erase(T const& value) {
		auto elem_iter = std::find(container.begin(), container.end(), value);
		if (elem_iter == container.end()) return false;
		container.erase(elem_iter);
		std::make_heap(container.begin(), container.end(), comp);
		return true;
	}
};