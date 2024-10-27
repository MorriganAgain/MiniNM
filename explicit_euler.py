import numpy as np


# ----<EXPLICIT EULER>---- #

# MAIN FUNCTION:
def explicit_euler(function, order, mesh_grid, initial_state) -> np.ndarray:
    
    if len(initial_state) != order + 1: 
        raise ValueError("Invalid initial state: should have same number of conditions as the order of the equation plue one for the independent variable.")
    
    solution = np.zeros([order + 1, np.size(mesh_grid, 0), ])  # Pre-allocate space for simulation results... 1 col per grid space, 1 row per order
    state = initial_state
    
    for step_count, step in enumerate(mesh_grid):
        solution[:, step_count] = state
        try: step_size = mesh_grid[step_count + 1] - step
        except: step_size = 0  # This is horrible... when implicit euler is implemented the last step can be calculated that way
        state = calculate_step(function, order, state, step_size)
        
    return solution

# SUB FUNCTIONS:
def calculate_step(function, order, state, step_size) -> np.ndarray:  # May be useful outside of the above functions
    
    new_state = np.zeros(np.shape(state))  # pre-allocate space for new state  
    
    for n in range(order + 1):
        if n == 0:
            new_state[n] = state[n] + step_size  # update independent variable 
        elif n == order:
            new_state[n] = state[n] + step_size * function(state)  # update final derivative 
        else:
            new_state[n] = state[n] + step_size * state[n+1]
        
    return new_state

