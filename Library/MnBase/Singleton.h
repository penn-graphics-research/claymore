#ifndef SINGLETON_H
#define SINGLETON_H

#include <assert.h>
// http ://www.boost.org/doc/libs/1_39_0/boost/pool/detail/singleton.hpp

namespace mn {

/*
 *	@note	Singleton
 */
// T must be: no-throw default constructible and no-throw destructible
template<typename T>
struct Singleton {
   public:
	static T& instance() {
		static T instance {};
		return instance;
	}
};

/*
 *	\class	ManagedSingleton
 *	\note	Used in global systems
 */
template<typename T>
struct ManagedSingleton {
   protected:
	static T* p_instance;//NOLINT(cppcoreguidelines-avoid-non-const-global-variables) Cannot avoid this

   public:
	static void startup() {
		assert(p_instance == nullptr);
		p_instance = new T();
	}
	static void shutdown() {
		assert(p_instance != nullptr);
		delete p_instance;
	}

	///
	T* operator->() {
		return p_instance;
	}

	static T& instance() {
		assert(p_instance != nullptr);
		return *p_instance;
	}
	static T* get_instance() {
		assert(p_instance != nullptr);
		return p_instance;
	}
	static const T* get_const_instance() {
		assert(p_instance != nullptr);
		return p_instance;
	}
};

template<typename T>
T* ManagedSingleton<T>::p_instance = nullptr;//NOLINT(cppcoreguidelines-avoid-non-const-global-variables) Cannot avoid this

}// namespace mn

#endif