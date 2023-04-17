#ifndef CPP_TIMERS_HPP
#define CPP_TIMERS_HPP

#include <fmt/color.h>
#include <fmt/core.h>

#include <chrono>

namespace mn {

struct CppTimer {
	using HRC		= std::chrono::high_resolution_clock;
	using NS		= std::chrono::nanoseconds;///< default timer unit
	using TimeStamp = HRC::time_point;

   private:
	TimeStamp last;
	TimeStamp cur;

   public:
	void tick() {
		last = HRC::now();
	}

	void tock() {
		cur = HRC::now();
	}

	float elapsed() {
		float duration = std::chrono::duration_cast<NS>(cur - last).count() * 1e-6;
		return duration;
	}

	void tock(std::string tag) {
		tock();
		fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag.c_str(), elapsed());
	}
};

}// namespace mn

#endif