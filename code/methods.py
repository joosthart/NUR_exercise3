import numpy as np

def romberg(f, start, end, m):
    """Romburg integration for a given function f from start to end. The oreder
    of the integration is given by m, i.e. the number of interval splits to use 
    to calculate the integral. 
 
    Args:
        f (func): Function taking as argument x coordinate and outputs 
            corresponding y
        start (int/float): Starting point for integration
        end (int/float): End point for integration
        m (int): Order of romburg integration
 
    Returns:
        float : Value of integral
    """
    # Initial stepsize
    h = end - start
 
    # Intialize array for estimates
    r = np.zeros(m)
    
#     print(f)
#     print('start:',start)
#     print('end:',end)
#     print('f(start):',f(start))
#     print('f(end):',f(end))
    # 0th oreder estimate
    r[0] = 0.5*h*(f(start)+f(end))
    
    Np = 1 # Number of new points
    for i in range(1, m-1):
        delta = h # step size between points
        h *= 0.5 # decreasing stepsize
        x = start + h
        for _ in range(Np):
            r[i] += f(x) # new evaluation
            x += delta
        
        # Combine new point with previously calculated 
        r[i] = 0.5*(r[i-1]+ delta*r[i])
        
        # increment number of points for next run
        Np*=2
    
    # Upgrading previous results by combining them. 
    Np = 1 # Reset number of points
    for i in range(1, m-1):
        Np *= 4
        for j in range(0,m-i):
            # combining results j and j+1 and storing it in j
            r[j] = (Np*r[j+1] - r[j])/(Np-1)
    
    return r[0]

def parabola_min(f, x1, x2, x3):
    y1,y2,y3 = f(x1), f(x2), f(x3)
    a = ((x3 -x2)*(y2-y1) + y3-y2)/(x3-x1)
    b = (y3-y2 + a*(x2-x3)) / (x3-x2)
    
    x = - b/(2*a)
    return x

def bracketing(f, a, b, w=1.618):
    """ Bracketing a minimum, using parabolic interpolation.
    Args:
        f (callable): Function for which to find root
        a (float): boundry of bracket
        b (float): boundry of bracket
        w (float, optional): splitting fraction of bracket. Defaults to 1.618.
    Returns:
        list/float: list of float containig bracket
    """
 
    # ensure that a < b
    if f(b) > f(a):
        a, b = b, a
    
    # make a guess for c
    c = b + (b-a)*w
    
    # if on the right hand side of b, retrun bracket [a,b,c]
    if f(c) > f(b):
        return [a, b, c]
    
    # find the minimum of the parabola throuh [a,b,c]
    d = parabola_min(f,a,b,c)
    
    # find out the order of the new bracket and return smallest bracket
    if f(d)<f(c):
        return [b, d, c]
    elif f(d)>f(b):
        return [a,b,d]
    # if d is to far from b, take section step
    elif abs(d-b) > 100*abs(c-b):
        d = c+(c-b)*w
        return [b,c,d]
    else:
        return[b,c,d]

def golden_section(f, xmin, xmax, target_acc=1e-6, maxit=1e4):
    """Finding the mimimum of a function, f, in the range [xmin, xmax] using the 
    Golden section algorithm.
    Args:
        f (callable): Function fo which to find minimum
        xmin (float): left boundry of bracket
        xmax (float): right boundry of bracket
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.
        maxit (int, optional): Maximum number of iterations. Defaults to 1e4.
    Returns:
        float: x-value of the obtained minimum
    """
    w = 0.38197 # 2-phi
    i = 0
    # Bracket the minimum using bracketing algorithm
    a,b,c = bracketing(f,xmin, xmax)
    
    while i < maxit:
        # Identify larger interval
        if abs(c-b) > abs(b-a):
            x1, x2 = b, c
        else:
            x1, x2 = a, b
 
        # Choose new point in a self similar way
        d = b + (x2 - x1)*w
 
        # abort if target acc reached and return best value
        if abs(c-a) < target_acc:
            if f(d) < f(b):
                return b
            else:
                return d
 
        # Tighten the bracket
        if f(d) < f(b):
            if x1 == b and x2 == c:
                a, b = b, d
            elif x1 == a and x2 == b:
                c, b = b, d
        else:
            if x1 == b and x2 == c:
                c = d
            elif x1 == a and x2 == b:
                a = d
        i+=1
 
    # if maxit reached, return last d
    return d