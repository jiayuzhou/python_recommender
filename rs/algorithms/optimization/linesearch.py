'''
Line search conditions:
- curvtrack: for non-monotone. 

Created on Feb 5, 2014

@author: jiayu.zhou
'''

from numpy.linalg import norm;
from numpy import isnan;
from numpy import abs;
from numpy import isinf;


#termination condition
FLAG_SUFFDESC = 1;
FLAG_TOLX     = 2;
FLAG_MAXFUNEV = 3;

def curvtrack(x, d, t, f_old, grad_g_x_d, smoothF, nonsmoothF, desc_param, x_tol, maxIter):
    #non-monotone curvature backtrack line search. 
    
    #initialization
    beta = 0.5; # line search parameter. 
    
    #main loop
    iternum = 0;
    while 1:
        iternum += 1;
        
        # trial point and function value. 
        [h_y, y] = nonsmoothF( x + t * d, t);
         
        [g_y, grad_g_y] = smoothF(y);
        
        f_y = g_y + h_y;
        
        desc = 0.5 * norm( y - x) **2 ;
        
        # check termination conditions. 
        if f_y < max(f_old) + desc_param * t * desc:
            flag = FLAG_SUFFDESC;
            break;
        elif t <= x_tol:
            flag = FLAG_TOLX;
            break;
        elif iternum >= maxIter:
            flag = FLAG_MAXFUNEV;
            break;
        
        # Backtrack if objective value not well-defined of function seems linear
        if isnan( f_y ) or isinf( f_y ) or abs(f_y - f_old[-1] - t * grad_g_x_d) <= 1e-9:
            t = beta * t;
        else: # safeguard quadratic interpolation.
            t_interp = -(grad_g_x_d * (t **2) ) / (2 * (f_y - f_old[-1] - t * grad_g_x_d));
            if 0.1 <= t_interp or t_interp <= 0.9 * t:
                t = t_interp;
            else:
                t = beta * t;
            
    return [y , f_y, grad_g_y, t, flag, iternum];
        
        
        