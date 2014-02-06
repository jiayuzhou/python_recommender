'''

Structured reconstruction by separable approximation (SpaRSA)

This implementation is the same as PNOPT. 

Created on Feb 5, 2014

@author: jiayu.zhou
'''

import numpy as np;
from numpy.linalg import norm;
from rs.algorithms.optimization.optimizer import Optimizer;
from rs.algorithms.optimization.linesearch import curvtrack;

class Opt_SpaRSA(Optimizer):
    '''
    Optimization algorithm: SpaRSA. 
    '''

    def __init__(self, backtrack_mem = 10, desc_param = 0.0001, \
                 max_fun_eval = 50000, max_iter = 5000, ftol = 1e-9, optim_tol = 1e-6, xtol = 1e-9,\
                  verbose = 0):
        '''
        Constructor
        '''
        self.backtrack_mem = backtrack_mem;
        self.desc_param    = desc_param;
        self.max_fun_eval  = max_fun_eval;
        self.max_iter      = max_iter;
        self.ftol          = ftol;
        self.optim_tol     = optim_tol;
        self.xtol          = xtol;
        self.verbose       = verbose;
        
    
    def optimize(self, smoothF, nonsmoothF, x):
        '''
        Optimize min(x) smoothF(x) + nonsmoothF(x)
        '''
        iter_num = 0;
        loop = 1;
        
        trace = {};
        trace['f_x']        = np.zeros(self.max_iter + 1);
        trace['fun_evals']  = np.zeros(self.max_iter + 1);
        trace['prox_evals'] = np.zeros(self.max_iter + 1);
        trace['optim']      = np.zeros(self.max_iter + 1);
        
        if self.verbose >0:
            print ' %4s   %6s  %6s  %12s  %12s  %12s \n' %\
                ('','Fun.', 'Prox', 'Step len.', 'Obj. val.', 'Optim.' );
        
        # function value at starting point. 
        [g_x, grad_g_x] = smoothF(x);
        h_x             = nonsmoothF(x);
        f_x             = g_x + h_x;
        
        # collect data for display/output
        fun_evals   = 1;
        prox_evals  = 0;
        [_, x_prox] = nonsmoothF(x - grad_g_x, 1);
        optim       = np.linalg.norm(x_prox - x, np.inf);
        
        trace['f_x'][1]        = f_x;
        trace['fun_evals'][1]  = fun_evals;
        trace['prox_evals'][1] = prox_evals;
        trace['optim'][1]      = optim;
        
        if self.verbose > 0:
            print ' %4d | %6d  %6d  %12s  %12.4e  %12.4e\n' \
                   % ( iter_num, fun_evals, prox_evals, '', f_x, optim );
        
        # optimality of the starting point.
        if optim <= self.optim_tol:
            flag = Optimizer.FLAG_OPTIM;
            msg  = Optimizer.MESSAGE_OPTIM;
            loop = 0;
        
        ## Main Loop
        
        # temporarily set variables. 
        x_old = x;
        grad_f_old = grad_g_x;
        f_old = np.zeros(0);
        
        while loop:
            iter_num += 1;
            
            # STEP1 search direction.
            if iter_num > 1:
                s = x - x_old;
                y = grad_g_x - grad_f_old;
                BBstep = (y.T * s)/(y.T * y);
                if BBstep <= 1e-9 or 1e9 <= BBstep:
                    BBstep = min(1, 1/norm(grad_g_x, 1));
            else:
                BBstep = min(1, 1/norm(grad_g_x, 1));
           
            # STEP2 line search.
            x_old = x;
            if iter_num + 1 > self.backtrack_mem:
                f_old = np.concatenate((f_old[1:], f_x));
            else:
                f_old[iter_num] = f_x;
            grad_f_old = grad_g_x;
            
            # line search. the ignored flag curvtrack_flag can be used to identify possible issues 
            #              in the line search procedure. 
            [x, f_x, grad_g_x, step, _, curvtrack_iters] = \
                curvtrack(x, - grad_g_x, BBstep, f_old, - norm(grad_g_x) **2, smoothF,\
                          nonsmoothF, self.desc_param,  self.xtol, self.max_fun_eval - fun_evals); 
            
            # update statistics for display/output/termination.
            fun_evals   = fun_evals  + curvtrack_iters;
            prox_evals  = prox_evals + curvtrack_iters; 
            [_, x_prox] = nonsmoothF(x - grad_g_x, 1);
            optim       = norm(x_prox - x, np.inf);
            
            if self.verbose > 0 and iter_num % self.verbose == 0: 
                print ' %4d | %6d  %6d  %12.4e  %12.4e  %12.4e'\
                     % (iter_num, fun_evals, prox_evals, step, f_x, optim);
            
            # stop condition
            if optim <= self.optim_tol:
                flag  = Optimizer.FLAG_OPTIM;
                msg   = Optimizer.MESSAGE_OPTIM;
                loop  = 0;
            elif norm(x - x_old, np.inf) / max(1, norm(x_old, np.inf)) <= self.xtol:
                flag  = Optimizer.FLAG_XTOL;
                msg   = Optimizer.MESSAGE_XTOL;
                loop  = 0;
            elif abs(f_old - f_x) / max(1, abs(f_old)) <= self.ftol:
                flag  = Optimizer.FLAG_FTOL;
                msg   = Optimizer.MESSAGE_XTOL;
            elif iter_num >= self.max_iter:
                flag  = Optimizer.FLAG_MAXITER;
                msg   = Optimizer.MESSAGE_MAXITER;
            elif fun_evals >= self.max_fun_eval:
                flag  = Optimizer.FLAG_MAXFEV;
                msg   = Optimizer.MESSAGE_MAXFEV;
                
        # clean-up trace (only show the first iter_num +1 ) before exit. 
        trace['f_x']        = trace['f_x'][0:iter_num+1];
        trace['fun_evals']  = trace['fun_evals'][0:iter_num+1];
        trace['prox_evals'] = trace['prox_evals'][0:iter_num+1];
        trace['optim']      = trace['optim'][0:iter_num+1];
                
        output = {'flag':flag, 'fun_evals':fun_evals, 'iters':iter_num, \
                  'optim':optim, 'prox_evals': prox_evals, 'trace': trace,
                  'msg': msg};
                  
        return [x, f_x, output];