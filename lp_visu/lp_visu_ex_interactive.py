"""A simple example that uses lp_visu with scipy.optimize.linprog to
show how the simplex algorithm runs."""

from lp_visu import LPVisu
from scipy.optimize import linprog

import numpy as np

print("click on the figure to show scipy.optimize.linprog Simplexe step")

# problem definition
A = [[1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
b = [8.0, 15.0, 18.0]
c = [4.0, 3.0]

x1_bounds = (0, None)
x2_bounds = (0, None)

# GUI bounds
x1_gui_bounds = (-1, 16)
x2_gui_bounds = (-1, 10)

visu = LPVisu(A, b, c,
              x1_bounds, x2_bounds,
              x1_gui_bounds, x2_gui_bounds)

def lp_simple_callback(optimizeResult):
    """A simple callback function to see what is happening to print each
    step of the algorithm and to use the visualization.

    """

    print("current iteration: " + str(optimizeResult["nit"]))
    print("current slack: " + str(optimizeResult["slack"]))
    print("current solution: " + str(optimizeResult["x"]))
    print()

    visu.draw_pivot_interactive(optimizeResult["x"], True)

# solve the problem
res = linprog(-1.0 * np.array(c), A_ub=A, b_ub=b,
              bounds=(x1_bounds, x2_bounds),
              callback=lp_simple_callback,
              options={"disp": True})

print(res)
