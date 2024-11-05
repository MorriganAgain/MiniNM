import numpy as np
import sys

def fixed_point(function, state, step_size, parameters, tolerance, limit) -> float:
    
    error = sys.maxsize
    current_guess = state.copy()
    iterations = 0
    
    while error > tolerance and iterations < limit:
        new_guess = state[-1] + step_size * function(current_guess, parameters)
        error = abs((new_guess - current_guess[-1]) / new_guess)
        
        current_guess[-1] = new_guess
        iterations += 1
        
    if iterations >= limit: 
        raise Exception("Exceeded iteration limit: Does not converge or converges too slowly.")
    
    return current_guess[-1]