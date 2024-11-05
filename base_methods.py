

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