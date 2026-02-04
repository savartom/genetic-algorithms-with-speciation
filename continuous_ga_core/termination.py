from continuous_ga_core.population import GAState
import time


def termination_by_generation(state: GAState, limit: int):
    return state.generation >= limit


def make_time_limiter():
    start_time = None

    def checker(state: GAState, limit: float):
        nonlocal start_time
        if start_time is None:
            start_time = time.monotonic()
            return False
        if (time.monotonic() - start_time) > limit:
            print(f"last generation: {state.generation}")
        return (time.monotonic() - start_time) > limit

    return checker
