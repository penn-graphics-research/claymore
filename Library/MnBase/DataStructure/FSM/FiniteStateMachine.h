#ifndef FINITE_STATE_MACHINE_H
#define FINITE_STATE_MACHINE_H
#include "MnBase/Meta/Optional.h"
#include "MnBase/Meta/Polymorphism.h"
#include "MnBase/Object/Property.h"

namespace mn {

/// CRTP
template<typename Derived, typename StateVariant>
class FSM {
   private:
	StateVariant state;

   public:
	explicit FSM(StateVariant&& state)
		: state(state) {}

	template<typename Event>
	void dispatch(Event&& event) {
		Derived& self = static_cast<Derived&>(*this);
		auto newState = match(state)([this, &event](auto& s) -> optional<StateVariant> {
			return self.onEvent(s, std::forward<Event>(event));
		});
		//[&](auto & s) ->optional<StateVariant> { return self.onEvent(s, (event)); });
		if(newState) {
			state = *std::move(newState);
		}
	}
};

}// namespace mn

#endif