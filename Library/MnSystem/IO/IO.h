#ifndef IO_H
#define IO_H
#include <tl/function_ref.hpp>

#include "MnBase/Concurrency/Concurrency.h"
#include "MnBase/Singleton.h"

namespace mn {

struct IO : Singleton<IO> {
   private:
	bool b_running;
	std::mutex mut;
	std::condition_variable cv;
	ThreadsafeQueue<std::function<void()>> jobs;
	std::thread th;

	void wait() {
		std::unique_lock<std::mutex> lk {mut};
		cv.wait(lk, [this]() {
			return !this->b_running || !this->jobs.empty();
		});
	};
	void worker() {
		while(b_running) {
			wait();
			auto job = jobs.try_pop();
			if(job) {
				(*job)();
			}
		}
	}

   public:
	IO()
		: b_running {true} {
		th = std::thread([this]() {
			this->worker();
		});
	}

	~IO() {
		while(!jobs.empty()) {
			cv.notify_all();
		}
		b_running = false;
		th.join();
	}

	//TODO: Maybe implement
	IO(IO& other)			  = delete;
	IO(IO&& other)			  = delete;
	IO& operator=(IO& other)  = delete;
	IO& operator=(IO&& other) = delete;

	static void flush() {
		while(!instance().jobs.empty()) {
			instance().cv.notify_all();
		}
	}
	static void insert_job(const std::function<void()>& job) {
		std::unique_lock<std::mutex> lk {instance().mut};
		instance().jobs.push(job);
		lk.unlock();
		instance().cv.notify_all();
	}
};

}// namespace mn

#endif