"""A module to visualize simplex algorithm for (integer) linear
programs with two variables.
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
                 x1_gui_bounds, x2_gui_bounds,
                 x1_grid_step=1, x2_grid_step=1,
                 epsilon=1E-6,
                 ilp = False,
                 A_cuts = None, b_cuts = None, integers = False,
                 notebook=False, xk=None, obj=None, scale=1.0):
        """Create a new LPVisu object.

        Keyword Arguments:
        A             -- a 2D matrix giving the constraints of the LP problem
        b             -- a vector representing the upper-bound of the constraints
        c             -- the coefficients of the linear function to be minimized
        x1_bounds     -- a pair representing x1 bounds. Use None for infinity
        x2_bounds     -- a pair representing x2 bounds. Use None for infinity
        x1_gui_bounds -- a pair representing x1 bounds in the GUI
        x2_gui_bounds -- a pair representing x2 bounds in the GUI
        x1_grid_step  -- an integer representing the step for x1 axis
        x2_grid_step  -- an integer representing the step for x2 axis
        epsilon       -- the precision needed for floating points operations.
                         Defaults to 1E-6
        ilp           -- is it an ILP problem (to draw integers)? Must be set to
                         True if cuts are to used.
                         Defaults to False
        notebook      -- is the object to be used in a Jupyter Notebook.
                         Defaults to False.
        xk            -- the coordinates of the pivot to plot when creating the
                         object (None if no drawing). Useful for notebooks. Must
                         be used with option notebook set to True
                         Defaults to None.
        obj           -- the value of the objective function if to be plotted when                          creating the object. Must be used with option notebook
                         set to True.
                         Defaults to None.
        scale         -- the scale factor for graphics. Must be used with option
                         notebook set to True.
                         Defaults to 1.0.
        """

        # attributes
        self.A = list(A)
        self.b = b
        self.c = c
        self.x1_bounds = x1_bounds
        self.x2_bounds = x2_bounds
        self.x1_gui_bounds = x1_gui_bounds
        self.x2_gui_bounds = x2_gui_bounds
        self.x1_grid_step = x1_grid_step
        self.x2_grid_step = x2_grid_step
        self.epsilon = epsilon
        self.notebook = notebook
        self.xk = xk
        self.obj = obj
        self.scale = scale
        if A_cuts:
            self.A_cuts = A_cuts
        else:
            self.A_cuts = []
        if b_cuts:
            self.b_cuts = b_cuts
        else:
            self.b_cuts = []
        self.integers = integers

        # prepare graphics objects
        self.ax = None
        self.pivot_patch = None
        self.obj_patch = None
        self.started = False
        self.lines = self._compute_lines(self.A, self.b)
        self.polygon, self.convex_hull = self._compute_polygon_convex_hull(self.A,
                                                                           self.b,
                                                                           self.lines)
        self.lines_cuts = []
        self.initial_patch = None
        self.cuts_patch = None
        self.cuts_lines_patch = []
        self.cuts_circles = []
        self.initial_polygon = np.array(self.polygon)
        self.initial_path = plt.Polygon([(self.initial_polygon[index, 0],
                                          self.initial_polygon[index, 1])
                                         for index in self.convex_hull.vertices],
                                        edgecolor='b', facecolor='cyan')

        # initialize picture
        self.__init_picture()

        # draw integers inside polygon if asked
        if integers:
            self.__draw_integers(self.initial_polygon, self.initial_path)

    def draw_objective_function(self, value):
        """Draw the objective function for a specific value.

        Keyword Arguments:
        value -- the value of the objective function. If None, remove objective function line.
        """

        if value is not None:
            points = [(self.x1_gui_bounds[0],
                       (value - self.x1_gui_bounds[0] * self.c[0]) /
                       self.c[1]),
                      (self.x1_gui_bounds[1],
                       (value - self.x1_gui_bounds[1] * self.c[0]) /
                       self.c[1])] \
                if abs(self.c[1]) > self.epsilon else \
                [(value / self.c[0], self.x2_gui_bounds[0]),
                 (value / self.c[0], self.x2_gui_bounds[1])]

            self.obj_patch = plt.Polygon(points, color='r', linewidth=2.0)
            self.ax.add_patch(self.obj_patch)
        else:
            if self.obj_patch is not None:
                self.obj_patch.remove()

    def draw_pivot(self, xk):
        """Draw a red circle at the current pivot position.

        Keyword Arguments:
        xk -- a pair representing the position of the new pivot. If None, remove pivot
        """

        if xk is not None:
            if self.pivot_patch is None:
                self.pivot_patch = plt.Circle((xk[0], xk[1]),
                                              0.25 if self.notebook else 0.1,
                                              fc='r')
            else:
                self.pivot_patch.center = (xk[0], xk[1])
            self.ax.add_patch(self.pivot_patch)
        else:
            if self.pivot_patch is not None:
                self.pivot_patch.remove()

    def draw_pivot_interactive(self, xk, key_pressed=False, wait_time=1):
        """Draw a red circle at the current pivot position.
        To be used interactively.

        Keyword Arguments:
        xk          -- a pair representing the position of the new pivot
        key_pressed -- True if a key or button must be pressed to continue else
                       wait for time seconds
        wait_time   -- the time in seconds to wait
        """

        if self.pivot_patch is None:
            self.pivot_patch = plt.Circle((0, 0), 0.1, fc='r')
        else:
            gui_line, = self.ax.plot([self.pivot_patch.center[0], xk[0]],
                                     [self.pivot_patch.center[1], xk[1]])
            gui_line.set_color('red')
            gui_line.set_linestyle('-')
            gui_line.set_linewidth(3)
            plt.draw()

        self.pivot_patch.center = (xk[0], xk[1])
        self.ax.add_patch(self.pivot_patch)

        if key_pressed:
            plt.waitforbuttonpress()
        else:
            plt.pause(wait_time)

    def add_cuts(self, A_cuts, b_cuts):
        """A method to add cuts.

        Keyword Arguments:
        A_cuts -- the A matrix for the cuts
        b_cuts -- the b matrix for the cuts
        """

        if self.cuts_patch is None:
            polygon, convex_hull = self._compute_polygon_convex_hull(self.A,
                                                                     self.b,
                                                                     self.lines)

            draw_polygon = np.array(polygon)
            self.initial_patch = plt.Polygon([(draw_polygon[index, 0], draw_polygon[index, 1])
                                              for index in convex_hull.vertices],
                                             edgecolor='r', facecolor='tomato')
            self.ax.add_patch(self.initial_patch)

        if self.cuts_patch is not None:
            self.cuts_patch.remove()

        for c in self.cuts_circles:
            c.remove()

        self.cuts_circles = []

        self.A_cuts = self.A_cuts + A_cuts
        self.b_cuts = self.b_cuts + b_cuts
        self.lines_cuts = self._compute_lines(
            self.A_cuts, self.b_cuts, bounds=False)

        polygon_cuts, convex_hull_cuts = self._compute_polygon_convex_hull(self.A + self.A_cuts,
                                                                           self.b + self.b_cuts,
                                                                           self.lines + self.lines_cuts)

        draw_polygon = np.array(polygon_cuts)
        self.cuts_patch = plt.Polygon([(draw_polygon[index, 0], draw_polygon[index, 1])
                                       for index in convex_hull_cuts.vertices],
                                      edgecolor='g', facecolor='palegreen')
        self.ax.add_patch(self.cuts_patch)

        for l in self.lines_cuts:
            line_patch = plt.Polygon(l, color='r', linewidth=2,
                                     linestyle='dashed', closed=False)
            self.ax.add_patch(line_patch)
            self.cuts_lines_patch.append(line_patch)

        self.__draw_integers(draw_polygon, self.cuts_patch)

    def reset_cuts(self):
        """Remove all cuts."""

        if self.initial_patch is not None:
            self.initial_patch.remove()
            self.initial_patch = None

        if self.cuts_patch is not None:
            self.cuts_patch.remove()

            for p in self.cuts_lines_patch:
                p.remove()

            self.A_cuts = []
            self.b_cuts = []
            self.lines_cuts = []
            self.cuts_patch = None

            self.__draw_integers(self.initial_polygon, self.initial_path)

    def _compute_lines(self, A, b, bounds=True):
        """Computes lines points for equations. Returns a list with points
        representing intersections of each constraint with the GUI bounds.

        This method is parametrized to be possibly used with subclasses.

        Keyword Arguments:
        A      -- the A matrix
        b      -- the b matrix
        bounds -- if x1 and x2 bounds should be taken into account (defaults: True)

        Not to be used outside the class.
        """

        lines = [[(self.x1_gui_bounds[0], (b[A.index(l)] - self.x1_gui_bounds[0] * l[0]) / l[1]),
                  (self.x1_gui_bounds[1], (b[A.index(l)] - self.x1_gui_bounds[1] * l[0]) / l[1])]
                 if l[1] != 0 else
                 [(b[A.index(l)] / l[0], self.x2_gui_bounds[0]),
                  (b[A.index(l)] / l[0], self.x2_gui_bounds[1])]
                 for l in A]

        if bounds:
            lines.append([(self.x1_bounds[0] if self.x1_bounds[0] is not None
                           else self.x1_gui_bounds[0], 0),
                          (self.x1_bounds[1] if self.x1_bounds[1] is not None
                           else self.x1_gui_bounds[1], 0)])

            lines.append([(0, self.x2_bounds[0] if self.x2_bounds[0] is not None else self.x2_gui_bounds[0]),
                          (0, self.x2_bounds[1] if self.x2_bounds[1] is not None else self.x2_gui_bounds[1])])

        return lines

    def _compute_polygon_convex_hull(self, A, b, lines):
        """Compute the polygon of admissible solutions and the associated
        convex hull. Returns a pair with first element being the list
        of points of the polygon and second element the convex hull.

        This method is parametrized to be possibly used with subclasses.

        Keyword Arguments:
        A     -- the A matrix
        b     -- the b matrix
        lines -- the GUI lines for the equations

        Not to be used outside the class.
        """

        # compute all intersections...
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
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
            if False in (np.dot(A_arr, p) - b <= self.epsilon):
                continue
            polygon.append(p)

        # compute convex hull
        convex_hull = ConvexHull(polygon)

        return polygon, convex_hull

    def __draw_equations_and_polygon(self, ax):
        """Draw equations of the linear programming problems and the
        associated polygon.

        Not to be used outside the class.
        """

        # draw lines for equations
        for line in self.lines:
            gui_line, = self.ax.plot([line[0][0], line[1][0]],
                                     [line[0][1], line[1][1]])
            gui_line.set_color('black')
            gui_line.set_linestyle('--')

        # draw polygon
        draw_polygon = np.array(self.polygon)
        self.ax.fill(draw_polygon[self.convex_hull.vertices, 0],
                     draw_polygon[self.convex_hull.vertices, 1],
                     facecolor='palegreen', edgecolor="g", lw=2)

    def __draw_integers(self, polygon, patch):
        """Internal function to draw integer points inside polygon

        Keyword Arguments:
        polygon -- the polygon into which draw integer points
        patch   -- the patch corresponding to the polygon
        """

        x1_min = min([p[0] for p in polygon])
        x1_max = max([p[0] for p in polygon])

        x2_min = min([p[1] for p in polygon])
        x2_max = max([p[1] for p in polygon])

        for x in range(int(x1_min), int(x1_max) + 1):
            for y in range(int(x2_min), int(x2_max) + 1):
                if patch._path.contains_point((x, y), radius=self.epsilon):
                    circle = plt.Circle((x, y), 0.075, fc='b')
                    self.cuts_circles.append(circle)
                    self.ax.add_patch(circle)

    def __init_picture(self):
        """Initialize the picture and draw the equations lines and polygon.

        Not to be used outside the class.
        """

        plt.ion()

        # create figure
        self.fig = plt.figure()

        if self.notebook:
            self.fig.set_size_inches(self.scale * self.x1_gui_bounds[1]-self.x1_gui_bounds[0],
                                     self.scale * self.x2_gui_bounds[1]-self.x2_gui_bounds[0])

        self.ax = plt.axes(xlim=self.x1_gui_bounds,
                           ylim=self.x2_gui_bounds)

        # set axes and grid
        self.ax.grid(color='grey', linestyle='-')
        self.ax.set_xticks(np.arange(self.x1_gui_bounds[0],
                                     self.x1_gui_bounds[1],
                                     self.x1_grid_step))
        self.ax.set_yticks(np.arange(self.x2_gui_bounds[0],
                                     self.x2_gui_bounds[1],
                                     self.x2_grid_step))
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')

        # draw equations and polygon
        self.__draw_equations_and_polygon(self.ax)

        # possibly draw pivot and objective function if option
        # notebook is True
        if self.notebook:
            # draw objective function if asked
            if self.obj:
                self.draw_objective_function(self.obj)

            # draw pivot if asked
            if self.xk is not None:
                self.draw_pivot(self.xk)

            # cuts if there are some
            if self.A_cuts is not None or self.b_cuts is not None:
                lines_cuts = self._compute_lines(self.A_cuts, self.b_cuts, bounds=False)

                polygon_cuts, convex_hull_cuts = self._compute_polygon_convex_hull(self.A + self.A_cuts,
                                                                                   self.b + self.b_cuts,
                                                                                   lines + lines_cuts)

                polygon_cuts = np.array(polygon_cuts)
                cuts_patch = plt.Polygon([(polygon_cuts[index, 0], polygon_cuts[index, 1])
                                          for index in convex_hull_cuts.vertices],
                                         edgecolor='b', facecolor='cyan')

                ax.add_patch(cuts_patch)

                for l in lines_cuts:
                    line_patch = plt.Polygon(l, color='b', linewidth=2,
                                             linestyle='dashed', closed=False)
                    ax.add_patch(line_patch)

            #draw integers if asked
            if self.integers is not None:
                if self.A_cuts is not None or self.b_cuts is not None:
                    polygon_integer = polygon_cuts
                    patch_integer = cuts_patch
                else:
                    polygon_integer = initial_polygon
                    patch_integer = initial_patch

                self.__draw_integers(ax, polygon_integer, patch_integer)


        # finalize
        plt.draw()
        plt.show()


class ILPVisu(LPVisu):
    """This class is a simple visualization for integer linear programs
    with 2 variables.

    """

    def __init__(self, A, b, c,
                 x1_bounds, x2_bounds,
                 x1_gui_bounds, x2_gui_bounds,
                 x1_grid_step=1, x2_grid_step=1,
                 epsilon=1E-6):
        """Create a new ILPVisu object.

        Keyword Arguments:
        A             -- a 2D matrix giving the constraints of the LP problem
        b             -- a vector representing the upper-bound of the constraints
        c             -- the coefficients of the linear function to be minimized
        x1_bounds     -- a pair representing x1 bounds. Use None for infinity
        x2_bounds     -- a pair representing x2 bounds. Use None for infinity
        x1_gui_bounds -- a pair representing x1 bounds in the GUI
        x2_gui_bounds -- a pair representing x2 bounds in the GUI
        x1_grid_step  -- an integer representing the step for x1 axis
        x2_grid_step  -- an integer representing the step for x2 axis
        epsilon       -- the precision needed for floating points operations.
                         Defaults to 1E-6
        """

        # call constructor of superclass
        super().__init__(A, b, c,
                         x1_bounds, x2_bounds,
                         x1_gui_bounds, x2_gui_bounds,
                         x1_grid_step, x2_grid_step,
                         epsilon)
        self.lines_cuts = []
        self.A_cuts = []
        self.b_cuts = []
        self.initial_patch = None
        self.cuts_patch = None
        self.cuts_lines_patch = []
        self.cuts_circles = []
        self.initial_polygon = np.array(self.polygon)
        self.initial_path = plt.Polygon([(self.initial_polygon[index, 0],
                                          self.initial_polygon[index, 1])
                                         for index in self.convex_hull.vertices],
                                        edgecolor='b', facecolor='cyan')
        self.__draw_integers(self.initial_polygon, self.initial_path)

    def add_cuts(self, A_cuts, b_cuts):
        """A method to add cuts.

        Keyword Arguments:
        A_cuts -- the A matrix for the cuts
        b_cuts -- the b matrix for the cuts
        """

        if self.cuts_patch is None:
            polygon, convex_hull = self._compute_polygon_convex_hull(self.A,
                                                                     self.b,
                                                                     self.lines)

            draw_polygon = np.array(polygon)
            self.initial_patch = plt.Polygon([(draw_polygon[index, 0], draw_polygon[index, 1])
                                              for index in convex_hull.vertices],
                                             edgecolor='r', facecolor='tomato')
            self.ax.add_patch(self.initial_patch)

        if self.cuts_patch is not None:
            self.cuts_patch.remove()

        for c in self.cuts_circles:
            c.remove()

        self.cuts_circles = []

        self.A_cuts = self.A_cuts + A_cuts
        self.b_cuts = self.b_cuts + b_cuts
        self.lines_cuts = self._compute_lines(
            self.A_cuts, self.b_cuts, bounds=False)

        polygon_cuts, convex_hull_cuts = self._compute_polygon_convex_hull(self.A + self.A_cuts,
                                                                           self.b + self.b_cuts,
                                                                           self.lines + self.lines_cuts)

        draw_polygon = np.array(polygon_cuts)
        self.cuts_patch = plt.Polygon([(draw_polygon[index, 0], draw_polygon[index, 1])
                                       for index in convex_hull_cuts.vertices],
                                      edgecolor='g', facecolor='palegreen')
        self.ax.add_patch(self.cuts_patch)

        for l in self.lines_cuts:
            line_patch = plt.Polygon(l, color='r', linewidth=2,
                                     linestyle='dashed', closed=False)
            self.ax.add_patch(line_patch)
            self.cuts_lines_patch.append(line_patch)

        self.__draw_integers(draw_polygon, self.cuts_patch)

    def reset_cuts(self):
        """Remove all cuts."""

        if self.initial_patch is not None:
            self.initial_patch.remove()
            self.initial_patch = None

        if self.cuts_patch is not None:
            self.cuts_patch.remove()

            for p in self.cuts_lines_patch:
                p.remove()

            self.A_cuts = []
            self.b_cuts = []
            self.lines_cuts = []
            self.cuts_patch = None

            self.__draw_integers(self.initial_polygon, self.initial_path)

    def __draw_integers(self, polygon, patch):
        """Internal function to draw integer points inside polygon

        Keyword Arguments:
        polygon -- the polygon into which draw integer points
        patch   -- the patch corresponding to the polygon
        """

        x1_min = min([p[0] for p in polygon])
        x1_max = max([p[0] for p in polygon])

        x2_min = min([p[1] for p in polygon])
        x2_max = max([p[1] for p in polygon])

        for x in range(int(x1_min), int(x1_max) + 1):
            for y in range(int(x2_min), int(x2_max) + 1):
                if patch._path.contains_point((x, y), radius=self.epsilon):
                    circle = plt.Circle((x, y), 0.075, fc='b')
                    self.cuts_circles.append(circle)
                    self.ax.add_patch(circle)

class LPVisuNB():
    """This class is a simple visualization for linear programs with 2
    variables to be used in a notebook with %matplotlib inline. Works
    with both LP and ILP problems.

    """

    def __init__(self, A=None, b=None, c=None,
                 x1_bounds=None, x2_bounds=None,
                 x1_gui_bounds=None, x2_gui_bounds=None,
                 x1_grid_step=1, x2_grid_step=1,
                 epsilon=1E-6,
                 xk=None, obj=None,
                 A_cuts=None, b_cuts=None,
                 integers = None,
                 original=None):
        """Create a new LPVisuNB object.

        Keyword Arguments:
        A             -- a 2D matrix giving the constraints of the LP problem
        b             -- a vector representing the upper-bound of the constraints
        c             -- the coefficients of the linear function to be minimized
        x1_bounds     -- a pair representing x1 bounds. Use None for infinity
        x2_bounds     -- a pair representing x2 bounds. Use None for infinity
        x1_gui_bounds -- a pair representing x1 bounds in the GUI
        x2_gui_bounds -- a pair representing x2 bounds in the GUI
        x1_grid_step  -- an integer representing the step for x1 axis
        x2_grid_step  -- an integer representing the step for x2 axis
        epsilon       -- the precision needed for floating points operations.
                         Defaults to 1E-6
        xk            -- the coordinates of the pivot to plot
        obj           -- the value for the objective function if to be plotted
        A_cuts        -- the A matrix for cuts
        b_cuts        -- the b matrix for cuts
        integers      -- not None if you want integers to be drawn inside polygon
        original      -- the original LPVisu object to start from
        """

        if original is not None:
            self.A = original.A
            self.b = original.b
            self.c = original.c
            self.x1_bounds = original.x1_bounds
            self.x2_bounds = original.x2_bounds
            self.x1_gui_bounds = original.x1_gui_bounds
            self.x2_gui_bounds = original.x2_gui_bounds
            self.x1_grid_step = original.x1_grid_step
            self.x2_grid_step = original.x2_grid_step
            self.epsilon = original.epsilon
        else:
            self.A = list(A)
            self.b = b
            self.c = c
            self.x1_bounds = x1_bounds
            self.x2_bounds = x2_bounds
            self.x1_gui_bounds = x1_gui_bounds
            self.x2_gui_bounds = x2_gui_bounds
            self.x1_grid_step = x1_grid_step
            self.x2_grid_step = x2_grid_step
            self.epsilon = epsilon

        self.xk = xk
        self.obj = obj
        self.A_cuts = A_cuts
        self.b_cuts = b_cuts
        self.integers = integers

        # draw picture
        self.__init_picture()

    def _compute_lines(self, A, b, bounds=True):
        """Computes lines points for equations. Returns a list with points
        representing intersections of each constraint with the GUI bounds.

        This method is parametrized to be possibly used with subclasses.

        Keyword Arguments:
        A      -- the A matrix
        b      -- the b matrix
        bounds -- if x1 and x2 bounds should be taken into account (defaults: True)

        Not to be used outside the class.
        """

        lines = [[(self.x1_gui_bounds[0], (b[A.index(l)] - self.x1_gui_bounds[0] * l[0]) / l[1]),
                  (self.x1_gui_bounds[1], (b[A.index(l)] - self.x1_gui_bounds[1] * l[0]) / l[1])]
                 if l[1] != 0 else
                 [(b[A.index(l)] / l[0], self.x2_gui_bounds[0]),
                  (b[A.index(l)] / l[0], self.x2_gui_bounds[1])]
                 for l in A]

        if bounds:
            lines.append([(self.x1_bounds[0] if self.x1_bounds[0] is not None
                           else self.x1_gui_bounds[0], 0),
                          (self.x1_bounds[1] if self.x1_bounds[1] is not None
                           else self.x1_gui_bounds[1], 0)])

            lines.append([(0, self.x2_bounds[0] if self.x2_bounds[0] is not None else self.x2_gui_bounds[0]),
                          (0, self.x2_bounds[1] if self.x2_bounds[1] is not None else self.x2_gui_bounds[1])])

        return lines

    def _compute_polygon_convex_hull(self, A, b, lines):
        """Compute the polygon of admissible solutions and the associated
        convex hull. Returns a pair with first element being the list
        of points of the polygon and second element the convex hull.

        This method is parametrized to be possibly used with subclasses.

        Keyword Arguments:
        A     -- the A matrix
        b     -- the b matrix
        lines -- the GUI lines for the equations

        Not to be used outside the class.
        """

        # compute all intersections...
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
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
            if False in (np.dot(A_arr, p) - b <= self.epsilon):
                continue
            polygon.append(p)

        # compute convex hull
        convex_hull = ConvexHull(polygon)

        return polygon, convex_hull

    def __draw_equations_and_polygon(self, ax, lines, polygon, convex_hull):
        """Draw equations of the linear programming problems and the
        associated polygon.

        Not to be used outside the class.
        """

        # draw lines for equations
        for line in lines:
            gui_line, = ax.plot([line[0][0], line[1][0]],
                                [line[0][1], line[1][1]])
            gui_line.set_color('black')
            gui_line.set_linestyle('--')

        # draw polygon
        draw_polygon = np.array(polygon)
        ax.fill(draw_polygon[convex_hull.vertices, 0],
                draw_polygon[convex_hull.vertices, 1],
                facecolor='palegreen', edgecolor="g", lw=2)

    def __draw_integers(self, ax, polygon, patch):
        """Internal function to draw integer points inside polygon

        Keyword Arguments:
        polygon -- the polygon into which draw integer points
        patch   -- the patch corresponding to the polygon
        """

        x1_min = min([p[0] for p in polygon])
        x1_max = max([p[0] for p in polygon])

        x2_min = min([p[1] for p in polygon])
        x2_max = max([p[1] for p in polygon])

        for x in range(int(x1_min), int(x1_max) + 1):
            for y in range(int(x2_min), int(x2_max) + 1):
                if patch._path.contains_point((x, y), radius=self.epsilon):
                    circle = plt.Circle((x, y), 0.075, fc='b')
                    ax.add_patch(circle)

    def __init_picture(self):
        """Initialize the picture and draw the equations lines, polygon, cuts
        and integers.

        Not to be used outside the class.

        """

        plt.ion()

        # create figure
        fig = plt.figure()
        fig.set_size_inches(self.x1_gui_bounds[1]-self.x1_gui_bounds[0],
                            self.x2_gui_bounds[1]-self.x2_gui_bounds[0])
        ax = plt.axes(xlim=self.x1_gui_bounds,
                      ylim=self.x2_gui_bounds)

        # set axes and grid
        ax.grid(color='grey', linestyle='-')
        ax.set_xticks(np.arange(self.x1_gui_bounds[0],
                                self.x1_gui_bounds[1],
                                self.x1_grid_step))
        ax.set_yticks(np.arange(self.x2_gui_bounds[0],
                                self.x2_gui_bounds[1],
                                self.x2_grid_step))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        # draw equations and polygon
        lines = self._compute_lines(self.A, self.b)
        polygon, convex_hull = self._compute_polygon_convex_hull(self.A,
                                                                 self.b,
                                                                 lines)
        self.__draw_equations_and_polygon(ax, lines, polygon, convex_hull)

        initial_polygon = np.array(polygon)
        initial_patch = plt.Polygon([(initial_polygon[index, 0],
                                      initial_polygon[index, 1])
                                     for index in convex_hull.vertices],
                                    edgecolor='r', facecolor='tomato')

        # cuts if there are some
        if self.A_cuts is not None or self.b_cuts is not None:
            lines_cuts = self._compute_lines(self.A_cuts, self.b_cuts, bounds=False)

            polygon_cuts, convex_hull_cuts = self._compute_polygon_convex_hull(self.A + self.A_cuts,
                                                                               self.b + self.b_cuts,
                                                                               lines + lines_cuts)

            polygon_cuts = np.array(polygon_cuts)
            cuts_patch = plt.Polygon([(polygon_cuts[index, 0], polygon_cuts[index, 1])
                                      for index in convex_hull_cuts.vertices],
                                     edgecolor='g', facecolor='palegreen')

            ax.add_patch(initial_patch)
            ax.add_patch(cuts_patch)

            for l in lines_cuts:
                line_patch = plt.Polygon(l, color='r', linewidth=2,
                                         linestyle='dashed', closed=False)
                ax.add_patch(line_patch)

        # draw integers if asked
        if self.integers is not None:
            if self.A_cuts is not None or self.b_cuts is not None:
                polygon_integer = polygon_cuts
                patch_integer = cuts_patch
            else:
                polygon_integer = initial_polygon
                patch_integer = initial_patch

            self.__draw_integers(ax, polygon_integer, patch_integer)

        # draw objective function if asked
        if self.obj is not None:
            points = [(self.x1_gui_bounds[0],
                       (self.obj - self.x1_gui_bounds[0] * self.c[0]) /
                       self.c[1]),
                      (self.x1_gui_bounds[1],
                       (self.obj - self.x1_gui_bounds[1] * self.c[0]) /
                       self.c[1])] \
                if abs(self.c[1]) > self.epsilon else \
                [(self.obj / self.c[0], self.x2_gui_bounds[0]),
                 (self.obj / self.c[0], self.x2_gui_bounds[1])]

            obj_patch = plt.Polygon(points, color='r', linewidth=2.0)
            ax.add_patch(obj_patch)

        # draw pivot if asked
        if self.xk is not None:
            pivot_patch = plt.Circle((self.xk[0], self.xk[1]),
                                                 0.1, fc='r')
            ax.add_patch(pivot_patch)

        # finalize
        plt.draw()
        plt.show()
