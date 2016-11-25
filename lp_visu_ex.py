"""A simple example that uses lp_visu to visualize the polygon of
acceptable solutions for a particular problem. The objective function
and pivot are also drawn with some (dummy) values.
"""

from lp_visu import LPVisu

# problem definition
A = [[1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
b = [8.0, 15.0, 18.0]
c = [-4.0, -3.0]

x1_bounds = (0, None)
x2_bounds = (0, None)

# GUI bounds
x1_gui_bounds = (-1, 16)
x2_gui_bounds = (-1, 10)

visu = LPVisu(A, b, c,
              x1_bounds, x2_bounds,
              x1_gui_bounds, x2_gui_bounds)

# draw objective function with value = 20
visu.draw_objective_function(20)
input()

# remove objective function
visu.draw_objective_function(None)
input()

# draw objective function with value = 20
visu.draw_objective_function(40)
input()

# add a pivot (badly placed!)
visu.draw_pivot((1, 1))
input()

# remove pivot
visu.draw_pivot(None)
input()

# add pivot again...
visu.draw_pivot((2, 2))
input()
