import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

from math import sqrt
from matplotlib import animation
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

def intersect_with_gui_bounds(coeffs, point):
    second_point = (point[0] + point[0] * coeffs[0],
                    point[1] + point[1] * coeffs[1])
    points = []

    try:
        points.append(intersect(x1_gui_bounds[0], x1_gui_bounds[1],
                                point, second_point))
    except:
        pass

    try:
        points.append(intersect(x1_gui_bounds[0], x2_gui_bounds[0],
                                point, second_point))
    except:
        pass

    try:
        points.append(intersect(x2_gui_bounds[0], x2_gui_bounds[1],
                                point, second_point))
    except:
        pass

    try:
        points.append(intersect(x1_gui_bounds[1], x2_gui_bounds[1],
                                point, second_point))
    except:
        pass

    return points

def draw_equations_and_polygon(ax):
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
    #ax.plot(my_poly[convex_hull.vertices, 0], my_poly[convex_hull.vertices, 1],
    #        'b-', lw=2)
    ax.fill(my_poly[convex_hull.vertices, 0], my_poly[convex_hull.vertices, 1],
            facecolor='DeepSkyBlue', edgecolor="b", lw=2)

X_POINT = None
Y_POINT = None
patch = None
STEP = 0.0
VECT = None
FIG = None
STARTED = False
ax = None
NB_IT = 0

def init_picture():
    global patch
    global objective
    global FIG
    global ax
    global STEP

    plt.ion()

    # create figure
    FIG = plt.figure()
    ax = plt.axes(xlim=x1_gui_bounds, ylim=x2_gui_bounds)
    patch = plt.Circle((x1_gui_bounds[0] - 1, x2_gui_bounds[0] - 1), 0.25, fc='y')

    STEP = max(x1_gui_bounds[1] - x1_gui_bounds[0], x2_gui_bounds[1] - x2_gui_bounds[0]) / 100

    # set axes and grid
    ax.grid(color='grey', linestyle='-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # draw equations and polygon
    draw_equations_and_polygon(ax)

    # finalize
    plt.draw()
    plt.show()

def animate(i):
    global patch
    global VECT
    global NB_IT
    global objective

    patch.center = (patch.center[0] + VECT[0] / NB_IT,
                    patch.center[1] + VECT[1] / NB_IT)

    return patch,

def init_anim():
    global ax
    global patch
    global objective

    ax.add_patch(patch)

    return patch,

def lp_simple_callback(xk, **kwargs):
    " a simple callback function to see what is happening..."
    global STEP
    global VECT
    global STARTED
    global ax
    global NB_IT

    print("current iteration: " + str(kwargs["nit"]))
    print("current tableau: \n" + str(kwargs["tableau"]))
    print("current indices: "   + str(kwargs["basis"]))
    print("current pivot: "     + str(kwargs["pivot"]))
    print("current solution: "  + str(xk))
    print()

    if STARTED:
        VECT = (xk[0] - patch.center[0],
                xk[1] - patch.center[1])
        dist = sqrt(VECT[0]**2 + VECT[1]**2)
        NB_IT = int(dist / STEP)

        if abs(VECT[0]) > 1E-6 or abs(VECT[1]) > 1E-6:
            anim = animation.FuncAnimation(FIG, animate, frames=NB_IT,
                                           init_func=init_anim,
                                           interval=100, blit=True, repeat=False)
    else:
        STARTED = True
        patch.center = (xk[0], xk[1])
        ax.add_patch(patch)

    input()

init_picture()

res = linprog(c, A_ub=A, b_ub=b, bounds = (x1_bounds, x2_bounds),
              callback=lp_simple_callback,
              options={"disp": True})

print(res)
