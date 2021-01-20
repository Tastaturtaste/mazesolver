#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> begin;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	std::string name;
public:
	Timer(std::string&& timername) : name(timername) {
		begin = std::chrono::high_resolution_clock::now();
	}
	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		auto duration = end - begin;
		std::cout << name << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms to run\n";
	}
	inline std::chrono::milliseconds get_duration() const { return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); }
};
