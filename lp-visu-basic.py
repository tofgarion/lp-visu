import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linprog
from scipy.spatial import ConvexHull

A = [[1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
b = [8.0, 15.0, 18.0]
c = [-4.0, -3.0]

x1_bounds     = (0, None)
x2_bounds     = (0, None)
x1_gui_bounds = (-1, 16)
x2_gui_bounds = (-1, 10)

# global variables to store domain vertices
LINES = []
POLYGON = []

def draw_equations_and_polygon(ax):
    def intersect(a1, a2, b1, b2):
        va = np.array(a2) - np.array(a1)
        vb = np.array(b2) - np.array(b1)
        vp = np.array(a1) - np.array(b1)

        vap = np.empty_like(va)
        vap[0] = -va[1]
        vap[1] = va[0]
        denom = np.dot(vap, vb)

        if abs(denom) < 1E-6:
            raise Exception('The two lines are parallel!')

        num = np.dot(vap, vp)

        return (num / denom.astype(float)) * vb + b1

    # compute lines
    lines = [[(x1_gui_bounds[0], (b[A.index(l)] - x1_gui_bounds[0] * l[0]) / l[1]),
              (x1_gui_bounds[1], (b[A.index(l)] - x1_gui_bounds[1] * l[0]) / l[1])]
             if l[1] != 0 else
             [(b[A.index(l)] / l[0], x2_gui_bounds[0]), (b[A.index(l)] / l[0], x2_gui_bounds[1])]
             for l in A]

    lines.append([(x1_bounds[0] if x1_bounds[0] is not None else x1_gui_bounds[0], 0),
                  (x1_bounds[1] if x1_bounds[1] is not None else x1_gui_bounds[1], 0)])

    lines.append([(0, x2_bounds[0] if x2_bounds[0] is not None else x2_gui_bounds[0]),
                  (0, x2_bounds[1] if x2_bounds[1] is not None else x2_gui_bounds[1])])

    for line in lines:
        gui_line, = ax.plot([line[0][0], line[1][0]],
                            [line[0][1], line[1][1]])
        gui_line.set_color('black')
        gui_line.set_linestyle('--')

    # compute all intersections...
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            try:
                intersections.append(intersect(lines[i][0],
                                               lines[i][1],
                                               lines[j][0],
                                               lines[j][1]))
            except Exception:
                pass

    # check which intersection is a vertex of the polygon
    # and build the polygon
    A_arr = np.array(A)
    polygon = []

    for p in intersections:
        if x1_bounds[0] is not None:
            if p[0] < x1_bounds[0]:
                continue
        if x1_bounds[1] is not None:
            if p[0] > x1_bounds[0]:
                continue
        if x2_bounds[0] is not None:
            if p[1] < x2_bounds[0]:
                continue
        if x2_bounds[1] is not None:
            if p[1] > x2_bounds[0]:
                continue
        if False in (np.dot(A_arr, p) <= b):
            continue
        polygon.append(p)

    # compute convex hull
    convex_hull = ConvexHull(polygon)

    # draw polygon
    my_poly = np.array(polygon)
    ax.plot(my_poly[convex_hull.vertices, 0], my_poly[convex_hull.vertices, 1],
            'r-', lw=2)
    ax.fill(my_poly[convex_hull.vertices, 0], my_poly[convex_hull.vertices, 1],
            facecolor='b', edgecolor="r", lw=2)

def init_picture():
    # create figure
    fig = plt.figure()
    ax = plt.axes(xlim=x1_gui_bounds, ylim=x2_gui_bounds)

    # set bounds
    #plt.xlim(x1_gui_bounds)
    #plt.ylim(x2_gui_bounds)

    # set axes and grid
    ax.grid(color='grey', linestyle='-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # draw equations and polygon
    draw_equations_and_polygon(ax)

    # finalize
    plt.draw()
    plt.show()

def lp_simple_callback(xk, **kwargs):
    " a simple callback function to see what is happening..."
    print("current iteration: " + str(kwargs["nit"]))
    print("current tableau: \n" + str(kwargs["tableau"]))
    print("current indices: "   + str(kwargs["basis"]))
    print("current pivot: "     + str(kwargs["pivot"]))
    print("current solution: "  + str(xk))
    print()
    input()

init_picture()

res = linprog(c, A_ub=A, b_ub=b, bounds = (x1_bounds, x2_bounds),
              callback=lp_simple_callback,
              options={"disp": True})

print(res)
