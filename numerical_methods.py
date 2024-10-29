import numpy as np


# ----<EXPLICIT EULER>---- #

# MAIN FUNCTION:
def explicit_euler(function, mesh, initial_state, parameters) -> np.ndarray:
    """ 
    Approximate an ODE over a given mesh using the explicit/forward euler method. 
    
    Arguments: 
    @ function (function): The ODE to be simulated, solved for the highest order derivative. This function should take in arguments state (array) and parameters (dict).
    @ mesh (array): The mesh over which to approximate the ODE. 
    @ initial_state (array): The initial values required to compute a solution. Should be organized like so: [x_i, y(x_i), y'(x_i),...]
    @ parameters (dict): Any constants or parameters that need to be passed into the specified function. Can be constant values or an arrays with the same length as the mesh.
    
    Returns:
    (array): Columns describe the state at each point in the mesh.
    """
    
    order = len(initial_state) - 1
    solution = np.zeros([order + 1, np.size(mesh, 0), ])  # Pre-allocate space for simulation results... 1 col per grid space, 1 row per order
    state = initial_state
    
    for step_count, step in enumerate(mesh):
        solution[:, step_count] = state
        try: step_size = mesh[step_count + 1] - step
        except: step_size = 0  # This is horrible... 
        
        state = calculate_explicit_step(function, order, state, step_size, pass_in_params(step_count, parameters))
        
    return solution

# SUB FUNCTIONS:
def pass_in_params(step_count, parameters) -> dict:
    
    """ 
    Reads through the parameters dictionary to determine how each parameter should be passed through to the calculate_step function.
    """
    
    passed_params = {}
    for key, val in parameters.items():
        if len(val) != 1:  # if the values vary with the indepenent variable
            passed_params[key] = val[step_count]
            
        else: passed_params[key] = val  # if constant
    return passed_params

def calculate_explicit_step(function, order, state, step_size, parameters) -> np.ndarray:  # May be useful outside of the above function
    
    """ 
    Approximate a given ODE over a single time step using the explicit/forward Euler method.
    
    Arguments: 
    @ function (function): The ODE to be simulated, solved for the highest order derivative. This function should take in arguments state (array) and parameters (dict).
    @ order (int): The order of the ODE.
    @ state (array): The current values. Should be organized like so: [x_i, y(x_i), y'(x_i),...]
    @ step_size: Specifies the finite difference to be used.
    @ parameters (dict): Any constants or parameters that need to be passed into the specified function. Can be constant values or an arrays with the same length as the mesh.
    
    Returns:
    (array): Describes the state over a single time step.
    """
    
    new_state = np.zeros(np.shape(state))  # pre-allocate space for new state  
    
    for n in range(order + 1):
        if n == 0:
            new_state[n] = state[n] + step_size  # update independent variable 
        elif n == order:
            new_state[n] = state[n] + step_size * function(state=state, parameters=parameters)  # update final derivative 
        else:
            new_state[n] = state[n] + step_size * state[n+1]
        
    return new_state





