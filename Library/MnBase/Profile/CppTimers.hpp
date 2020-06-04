#ifndef __CPP_TIMERS_HPP_
#define __CPP_TIMERS_HPP_

#include <chrono>
#include <fmt/color.h>
#include <fmt/core.h>

namespace mn {

struct CppTimer {
  using HRC = std::chrono::high_resolution_clock;
  using NS = std::chrono::nanoseconds; ///< default timer unit
  using TimeStamp = HRC::time_point;

  void tick() { last = HRC::now(); }
  void tock() { cur = HRC::now(); }
  float elapsed() {
    float duration = std::chrono::duration_cast<NS>(cur - last).count() * 1e-6;
    return duration;
  }
  void tock(std::string tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag.c_str(), elapsed());
  }

private:
  TimeStamp last, cur;
};

} // namespace mn

#endif