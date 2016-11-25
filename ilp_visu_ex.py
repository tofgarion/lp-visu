"""A simple example that uses lp_visu to visualize the polygon of
acceptable solutions for a particular integer linear programming.
"""

from lp_visu import ILPVisu

# problem definition
A = [[1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
b = [8.0, 15.0, 18.0]
c = [-4.0, -3.0]

x1_bounds = (0, None)
x2_bounds = (0, None)

# GUI bounds
x1_gui_bounds = (-1, 16)
x2_gui_bounds = (-1, 10)

visu = ILPVisu(A, b, c,
               x1_bounds, x2_bounds,
               x1_gui_bounds, x2_gui_bounds)
input()

# adding some cuts
visu.add_cuts([[1.0, 1.0], [1.0, 0.0]],
              [10.0, 7.0])
input()

# add one cut
visu.add_cuts([[0.0, 1.0]],
              [6.0])
input()

# reset cuts and add only one
visu.reset_cuts()
input()

visu.add_cuts([[0.0, 1.0]],
              [6.0])
input()
