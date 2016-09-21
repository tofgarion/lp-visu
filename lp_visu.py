"""A module to visualize simplex algorithm for linear programs with
two variables.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import ConvexHull

def intersect(a1, a2, b1, b2):
    """
    Helper function to compute intersection of the two lines (a1, a2)
    and (b1, b2). An exception will be raised if the two lines are
    parallel.

    Keyword Arguments:
    a1 -- a pair representing the first point of the first line
    a2 -- a pair representing the secon point of the first line
    b1 -- a pair representing the first point of the second line
    b2 -- a pair representing the second point of the second line

    """

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


class LPVisu:
    """This class is a simple visualization for simplex resolution for
    linear programs with 2 variables.
    """

    def __init__(self, A, b, c,
                 x1_bounds, x2_bounds,
                 x1_gui_bounds, x2_gui_bounds):
        """Create a new LPVisu object.

        Keyword Arguments:
        A             -- a 2D matrix giving the constraints of the LP problem
        b             -- a vector representing the upper-bound of the constraints
        c             -- the coefficients of the linear function to be minimized
        x1_bounds     -- a pair representing x1 bounds. Use None for infinity
        x2_bounds     -- a pair representing x2 bounds. Use None for infinity
        x1_gui_bounds -- a pair representing x1 bounds in the GUI
        x2_gui_bounds -- a pair representing x2 bounds in the GUI

        """

        self.A             = A
        self.b             = b
        self.c             = c
        self.x1_bounds     = x1_bounds
        self.x2_bounds     = x2_bounds
        self.x1_gui_bounds = x1_gui_bounds
        self.x2_gui_bounds = x2_gui_bounds
        self.ax            = None
        self.patch         = None
        self.started       = False

        # initialize picture
        self.__init_picture()

    def draw_pivot(self, xk, key_pressed=False, wait_time=1):
        """Draw a yellow circle at the current pivot position.

        Keyword Arguments:
        xk          -- a pair representing the position of the new pivot
        key_pressed -- True if a key or button must be pressed to continue else
                       wait for time seconds
        wait_time   -- the time in seconds to wait
        """

        # draw current pivot and current equation
        if self.started:
            if self.patch is None:
                self.patch = plt.Circle((self.x1_gui_bounds[0] - 1,
                                         self.x2_gui_bounds[0] - 1),
                                        0.25, fc='r')
            else:
                gui_line, = self.ax.plot([self.patch.center[0], xk[0]],
                                         [self.patch.center[1], xk[1]])
                gui_line.set_color('red')
                gui_line.set_linestyle('-')
                gui_line.set_linewidth(3)
                plt.draw()

            self.patch.center = (xk[0], xk[1])
            self.ax.add_patch(self.patch)
        else:
            self.started      = True

        if key_pressed:
            plt.waitforbuttonpress()
        else:
            plt.pause(wait_time)


    def __intersect_with_gui_bounds(self, coeffs, point):
        """Compute intersection points between a line represented by its coeff
        and one of its point and the GUI bounds.

        Not to be used outside the class.

        """
        second_point = (point[0] + point[0] * coeffs[0],
                        point[1] + point[1] * coeffs[1])
        points = []

        try:
            points.append(intersect(self.x1_gui_bounds[0], self.x1_gui_bounds[1],
                                    point, second_point))
        except:
            pass

        try:
            points.append(intersect(self.x1_gui_bounds[0], self.x2_gui_bounds[0],
                                    point, second_point))
        except:
            pass

        try:
            points.append(intersect(self.x2_gui_bounds[0], self.x2_gui_bounds[1],
                                    point, second_point))
        except:
            pass

        try:
            points.append(intersect(self.x1_gui_bounds[1], self.x2_gui_bounds[1],
                                    point, second_point))
        except:
            pass

        return points

    def __draw_equations_and_polygon(self, ax):
        """Draw equations of the linear programming problems and the
        associated polygon.

        Not to be used outside the class.

        """

        # compute lines for equations
        lines = [[(self.x1_gui_bounds[0], (self.b[self.A.index(l)] - self.x1_gui_bounds[0] * l[0]) / l[1]),
                  (self.x1_gui_bounds[1], (self.b[self.A.index(l)] - self.x1_gui_bounds[1] * l[0]) / l[1])]
                 if l[1] != 0 else
                 [(self.b[self.A.index(l)] / l[0], self.x2_gui_bounds[0]),
                  (self.b[self.A.index(l)] / l[0], self.x2_gui_bounds[1])]
                 for l in self.A]

        lines.append([(self.x1_bounds[0] if self.x1_bounds[0] is not None
                       else self.x1_gui_bounds[0], 0),
                      (self.x1_bounds[1] if self.x1_bounds[1] is not None
                       else self.x1_gui_bounds[1], 0)])

        lines.append([(0, self.x2_bounds[0] if self.x2_bounds[0] is not None else self.x2_gui_bounds[0]),
                      (0, self.x2_bounds[1] if self.x2_bounds[1] is not None else self.x2_gui_bounds[1])])

        for line in lines:
            gui_line, = self.ax.plot([line[0][0], line[1][0]],
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
        A_arr = np.array(self.A)
        polygon = []

        for p in intersections:
            if self.x1_bounds[0] is not None:
                if p[0] < self.x1_bounds[0]:
                    continue
            if self.x1_bounds[1] is not None:
                if p[0] > self.x1_bounds[0]:
                    continue
            if self.x2_bounds[0] is not None:
                if p[1] < self.x2_bounds[0]:
                    continue
            if self.x2_bounds[1] is not None:
                if p[1] > self.x2_bounds[0]:
                    continue
            if False in (np.dot(A_arr, p) <= self.b):
                continue
            polygon.append(p)

        # compute convex hull
        convex_hull = ConvexHull(polygon)

        # draw polygon
        my_poly = np.array(polygon)
        self.ax.fill(my_poly[convex_hull.vertices, 0], my_poly[convex_hull.vertices, 1],
                     facecolor='palegreen', edgecolor="g", lw=2)

    def __init_picture(self):
        """Initialize the picture and draw the equations lines and polygon.

        Not to be used outside the class.
        """

        plt.ion()

        # create figure
        self.fig   = plt.figure()
        self.ax    = plt.axes(xlim=self.x1_gui_bounds,
                              ylim=self.x2_gui_bounds)

        # set axes and grid
        self.ax.grid(color='grey', linestyle='-')
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')

        # draw equations and polygon
        self.__draw_equations_and_polygon(self.ax)

        # finalize
        plt.draw()
        plt.show()
