#ifndef CONCURRENCY_H
#define CONCURRENCY_H

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "MnBase/Meta/Optional.h"

namespace mn {

template<typename F, typename... Ts>
inline auto really_async(F&& f, Ts&&... params) {
	return std::async(std::launch::async, std::forward<F>(f), std::forward<Ts>(params)...);
}

/// <<C++ concurrency in action>>
template<typename T>
class ThreadsafeQueue {
   private:
	mutable std::mutex mut;
	std::queue<T> data_queue;
	std::condition_variable data_cond;

   public:
	ThreadsafeQueue()  = default;
	~ThreadsafeQueue() = default;

	ThreadsafeQueue(ThreadsafeQueue const& other) noexcept {
		std::lock_guard<std::mutex> lk(other.mut);
		data_queue = other.data_queue;
	}

	ThreadsafeQueue(ThreadsafeQueue&& other) noexcept {
		std::lock_guard<std::mutex> lk(other.mut);
		data_queue = std::move(other.data_queue);
	}

	ThreadsafeQueue& operator=(ThreadsafeQueue const& other) noexcept {
		if(&other != this) {
			std::lock_guard<std::mutex> lk(other.mut);
			data_queue = other.data_queue;
		}

		return *this;
	}

	ThreadsafeQueue& operator=(ThreadsafeQueue&& other) noexcept {
		if(&other != this) {
			std::lock_guard<std::mutex> lk(other.mut);
			data_queue = std::move(other.data_queue);
		}

		return *this;
	}

	void push(T new_value) {
		std::lock_guard<std::mutex> lk(mut);
		data_queue.push(new_value);
		data_cond.notify_one();
	}

	void wait_and_pop(T& value) {
		/// spinlock should be better
		std::unique_lock<std::mutex> lk(mut);
		data_cond.wait(lk, [this] {
			return !data_queue.empty();
		});
		value = data_queue.front();
		data_queue.pop();
	}

	decltype(auto) wait_and_pop() {
		/// spinlock should be better
		std::unique_lock<std::mutex> lk(mut);
		data_cond.wait(lk, [this] {
			return !data_queue.empty();
		});
		T value = data_queue.front();
		data_queue.pop();
		return value;
	}

	std::shared_ptr<T> wait_and_pop_ptr() {
		std::unique_lock<std::mutex> lk(mut);
		data_cond.wait(lk, [this] {
			return !data_queue.empty();
		});
		std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
		data_queue.pop();
		return res;
	}

	bool try_pop(T& value) {
		std::lock_guard<std::mutex> lk(mut);
		if(data_queue.empty()) {
			return false;
		}
		value = data_queue.front();
		data_queue.pop();
		return true;
	}

	decltype(auto) try_pop() {
		std::lock_guard<std::mutex> lk(mut);
		if(data_queue.empty()) {
			return optional<T> {};
		}
		optional<T> value {data_queue.front()};
		data_queue.pop();
		return value;
	}

	std::shared_ptr<T> try_pop_ptr() {
		std::lock_guard<std::mutex> lk(mut);
		if(data_queue.empty()) {
			return std::shared_ptr<T>();
		}
		std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
		data_queue.pop();
		return res;
	}

	bool empty() const {
		std::lock_guard<std::mutex> lk(mut);
		return data_queue.empty();
	}
};

}// namespace mn

#endif