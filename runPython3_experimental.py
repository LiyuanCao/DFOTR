"""This file executes the derivative free optimization (DFO) solver to 
minimize a blackbox function.
 
User is required to import a blackbox function, and provide a 
starting point. User can overwrite the default algorithm and/or the 
default parameters used in the solver. 
"""
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"Python3")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
# import solver
# from Python3.dfo_tr import dfo_tr
from DFOTR2x4.dfo_tr import dfo_tr
# import blackbox functions
from funcs_def import arwhead, rosen


# choose function
a = -9.
b = 1.5
def simple_quartic(x):
    # The arwhead function
    x = (x - a) / b
    res = (x**2 + x[-1]**2)**2 - 4*x + 3
    res = sum( res ) - res[-1]
    return res
print(simple_quartic(np.array([a+b, a])))
func = simple_quartic
# func = arwhead
# func = rosen

# starting point
n = 6
x0 = np.ones(n) * 5.0
# x0 = np.repeat(np.array([[-1.2, 1]]), 2, axis=0).flatten()

# bounds
# bounds = None
bounds = [(-10,10)]*n

# overwrite default settings
customOptions = {'alg_model': 'quadratic',
                'alg_TR': 'box', 
                'alg_TRsub': 'exact',
                'alg_feval': 4,
                'tr_delta': 5, 
                'sample_initial': 2, 
                'sample_toremove': 4.0,
                'stop_nfeval': 128,
                'stop_predict': 0.,
                'verbosity': 2
                }
for i in range(18,1000):
    customOptions['rng_seed'] = i
    nAsk = 8 // customOptions['alg_feval']

    hist_x = np.empty((0,n))
    hist_fx = np.empty((0, customOptions['alg_feval']))
    hist_phix = np.array([])
    hist_center = []
    hist_delta = []
    # optimization in ask and tell frameword
    optimizer = dfo_tr(x0, bounds=bounds, options=customOptions)
    for i in range(16):
        x = optimizer.ask(nAsk)
        fx = [func(xi) + 3000.0 * np.random.randn() for xi in x]
        for j in range(nAsk):
            hist_x = np.vstack((hist_x, x[j*customOptions['alg_feval']]))
            hist_fx = np.vstack((hist_fx, fx[j*customOptions['alg_feval']:(j+1)*customOptions['alg_feval']]))
            hist_phix = np.append(hist_phix, func(x[j*customOptions['alg_feval']]))
        hist_center.append(optimizer.model.center)
        hist_delta.append(optimizer.model.delta)
        optimizer.tell(x,fx)
        if optimizer._stop():
            break

    idx = np.nanargmin(hist_phix)
    x = hist_x[idx]
    fx = hist_phix[idx]
    # print result
    print("Printing result for function " + func.__name__ + ":")
    print("best point: {}, with obj: {:.6f}".format(
        np.around(x, decimals=5), float(fx)))

########################################################################
########################################################################
# ## Make plots. 
# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation

# # plot the results 
# plt.rcParams['figure.figsize'] = [10,7]
# plt.scatter(np.repeat(range(len(hist_phix)), customOptions['alg_feval']),\
#     hist_fx.flatten(), c='g', alpha=0.1, label='function evalues')
# plt.scatter(range(len(hist_phix)), hist_phix, marker='o', c='k', alpha=1.0, label='true values')
# # plt.plot(range(len(history_f)), np.array(list(accumulate(true_values, min))))
# plt.legend()
# plt.xlabel('function evaluation in chronological order')
# plt.ylabel('$f(x)$')
# plt.show()


# fig, ax = plt.subplots()
# ## heatmap of the function 
# lb = [-10, -10]
# ub = [10,10]
# x1 = np.linspace(lb[0],ub[0],100)
# x2 = np.linspace(lb[1],ub[1],100)
# ### filling the heatmap, value by value
# fun_map = np.empty((x1.size, x2.size))
# for i in range(x1.size):
#     for j in range(x2.size):
#         fun_map[i,j] = func(np.array([x1[i], x2[j]]))
# ### plot heatmap
# im = ax.imshow(
#     fun_map,
#     extent=(lb[0], ub[0], lb[1], ub[1]),
#     origin='lower')
# fig.colorbar(im)

# scat = ax.scatter([], [], c='w', alpha=0.4, label='current')
# scat1 = ax.scatter([], [], c='r', alpha=0.4, label='current')
# line, = ax.plot([], [], lw=2)

# def setup_plot():
#     """Initial drawing of the scatter plot."""
#     ax.axis([lb[0], ub[0], lb[1], ub[1]])
#     scat = ax.scatter([], [], c='w', alpha=0.4, label='current')
#     scat1 = ax.scatter([], [], c='r', alpha=0.4, label='current')
#     line.set_data([], [])

#     # For FuncAnimation's sake, we need to return the artist we'll be using
#     # Note that it expects a sequence of artists, thus the trailing comma.
#     return scat, scat1, line, 

# def update(i):
#     """Update the scatter plot."""
#     data = np.array(hist_x)[:nAsk*(i+1),:2]
#     scat.set_offsets(data)

#     data1 = np.array(hist_x)[nAsk*i:nAsk*(i+1),:2]
#     scat1.set_offsets(data1)

#     if i > 0:
#         c = hist_center[i]
#         delta = hist_delta[i]
#         dataTR = np.array([ [c[0] - delta, c[1] + delta],
#                             [c[0] + delta, c[1] + delta],
#                             [c[0] + delta, c[1] - delta],
#                             [c[0] - delta, c[1] - delta],
#                             [c[0] - delta, c[1] + delta]
#         ])
#         line.set_data(dataTR[:,0], dataTR[:,1])

#     # We need to return the updated artist for FuncAnimation to draw..
#     # Note that it expects a sequence of artists, thus the trailing comma.
#     return scat, scat1, line, 

# ani = animation.FuncAnimation(fig, update, interval=400, frames=16, init_func=setup_plot, blit=True)
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('DFOTR.gif', writer='writer')
# plt.show()