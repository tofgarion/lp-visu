import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linprog
from scipy.spatial import ConvexHull

A = [[1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
b = [8.0, 15.0, 18.0]
c = [-4.0, -3.0]

x1_bounds     = (0, None)
x2_bounds     = (0, None)
x1_gui_bounds = (-2, 16)
x2_gui_bounds = (-2, 10)

LINES = [ [(x1_gui_bounds[0], (b[A.index(l)] - x1_gui_bounds[0] * l[0]) / l[1]),
           (x1_gui_bounds[1], (b[A.index(l)] - x1_gui_bounds[1] * l[0]) / l[1])]
          if not (l[1] == 0) else [(b[A.index(l)] / l[0], x2_gui_bounds[0]), (b[A.index(l)] / l[0], x2_gui_bounds[1])]
          for l in A ]

LINES.append([(x1_bounds[0] if not (x1_bounds[0] is None) else x1_gui_bounds[0], 0),
              (x1_bounds[1] if not (x1_bounds[1] is None) else x1_gui_bounds[1], 0)])

LINES.append([(0, x2_bounds[0] if not (x2_bounds[0] is None) else x2_gui_bounds[0]),
              (0, x2_bounds[1] if not (x2_bounds[1] is None) else x2_gui_bounds[1])])

def draw_lines():
    for l in LINES:
        line_2d, = plt.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]])
        line_2d.set_color('black')
        line_2d.set_linestyle('--')

# compute all intersections...
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

INTERSECTIONS = []

for i in range(len(LINES)):
    for j in range(i+1, len(LINES)):
        try:
            INTERSECTIONS.append(intersect(LINES[i][0],
                                           LINES[i][1],
                                           LINES[j][0],
                                           LINES[j][1]))
        except Exception:
            pass

# check which intersection is a vertex of the polygon
POLYGON = []
A_arr = np.array(A)

for p in INTERSECTIONS:
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
    POLYGON.append(p)

# compute convex hull
CONVEX_HULL = ConvexHull(POLYGON)

print(INTERSECTIONS)
print(POLYGON)
print([POLYGON[i] for i in CONVEX_HULL.vertices])

def draw_polygon():
    my_poly = np.array(POLYGON)
    line_2d, = plt.plot(my_poly[CONVEX_HULL.vertices, 0], my_poly[CONVEX_HULL.vertices, 1], 'r-', lw=2)
    line_2d, = plt.fill(my_poly[CONVEX_HULL.vertices, 0], my_poly[CONVEX_HULL.vertices, 1], facecolor='b', edgecolor="r", lw=2)

def lp_simple_callback(xk, **kwargs):
    " a simple callback function to see what is happening..."
    print("current iteration: " + str(kwargs["nit"]))
    print("current tableau: \n" + str(kwargs["tableau"]))
    print("current indices: "   + str(kwargs["basis"]))
    print("current pivot: "     + str(kwargs["pivot"]))
    print("current solution: "  + str(xk))
    print()
    input()

plt.ylim(x2_gui_bounds)
draw_lines()
draw_polygon()
plt.draw()
plt.show()

res = linprog(c, A_ub=A, b_ub=b, bounds = (x1_bounds, x2_bounds),
              callback=lp_simple_callback,
              options={"disp": True})

print(res)
