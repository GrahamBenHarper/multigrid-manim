from manim import *
import numpy as np
import math
import random

def create1dLaplaceMatrix(n: int, bcs: bool = True) -> np.array:
  """
  Create a Laplacian matrix corresponding to the 1D (1/h^2)*[-1,2,-1] stencil.
  @param n determines the size of the n-by-n matrix A
  @param bcs determines if the first and last row are assigned boundary conditions
  Note: For a [0,L] domain, h=L/(n-1), so one should modify A = (1./L**2)*A
  """
  # assume h=1./(n-1) for the problem.
  h = 1./(n-1)
  A = (1./(h**2))*(np.diag(2*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1))
  # if bcs is true, zero out the first and last rows and set the diagonal to 1.
  if bcs:
    A[0,:] = np.zeros(n)
    A[0,0] = 1
    A[n-1,:] = np.zeros(n)
    A[n-1,n-1] = 1
  return A

def createSineVector(coordinates: np.array) -> np.array:
  return np.sin(coordinates)

def createRandomVector(coordinates: np.array) -> np.array:
  return np.random.rand(len(coordinates))

# apply the Jacobi smoother x = x + omega*Dinv(b-Ax)
def applyJacobi(x: np.array, b: np.array, A: np.array, omega: float) -> np.array:
  """
  Apply a Jacobi smoother to the input array x based on the matrix A, rhs b, weight omega.
  @param x The input vector
  @param b The right-hand side of the equation being solved, Ax=b
  @param A The matrix for the equation being solved, Ax=b
  @param omega The weighting parameter for the update
  """
  Dinv = np.diag(1./np.diag(A))
  r = b - np.matmul(A,x)
  return x + omega*np.matmul(Dinv,r)

# apply the Gauss-Seidel smoother x = x + omega*Linv(b-Ax)
def applyGaussSeidel(x: np.array, b: np.array, A: np.array, omega: float) -> np.array:
  """
  Apply a Gauss-Seidel smoother to the input array x based on the matrix A, rhs b, weight omega.
  @param x The input vector
  @param b The right-hand side of the equation being solved, Ax=b
  @param A The matrix for the equation being solved, Ax=b
  @param omega The weighting parameter for the update
  """
  Linv = np.linalg.inv(np.tril(A))
  r = b - np.matmul(A,x)
  return x + omega*np.matmul(Linv,r)

# apply a linear interpolation restrictor matrix-free
# Briggs uses a factor of 1/4 for R
def applyLinearInterpR(x: np.array) -> np.array:
  """
  Apply a 1D linear interpolation R to a vector x, from Briggs
  """
  if(len(x)%2==0):
    print("Attempting to apply LinearInterpP on a vector with {} entries! Only odd numbers of entries are supported!".format(len(x)))
  
  scale = 4. #2.*math.sqrt(2.)
  xout = np.zeros(int((len(x)-1)/2))
  for i in range(0,len(xout)):
    xout[i] = (x[2*i] + 2.*x[2*i+1] + x[2*i+2])/scale
  
  return xout

# apply a linear interpolation prolongator matrix-free
# Briggs uses a factor of 1/2 for P
def applyLinearInterpP(x: np.array) -> np.array:
  """
  Apply a 1D linear interpolation P to a vector x, from Briggs
  """
  xout = np.zeros(2*len(x)+1)

  scale = 2. #2.*math.sqrt(2.)
  xout[0] = x[0]/scale
  for i in range(0,len(x)-1):
    xout[2*i+1] = 2*x[i]/scale
    xout[2*i+2] = (x[i]+x[i+1])/scale
  
  return xout

# show how Jacobi affects the breakdown of frequency
class JacobiFrequency(Scene):
  def construct(self):
#    title = MathTex(r"\mbox{Solve} Ax=b").to_edge(UP)
    title = Text("Jacobi's Method - Laplace's Equation").scale(0.8).to_edge(UP)
    jacobitext = MathTex(r"\mathbf{x} = \mathbf{x} + \omega D^{-1}(\mathbf{b}-A\mathbf{x})").scale(0.8).shift(2.5*UP)
    smoothtext = Text("Low Frequency Mode").scale(0.5).to_edge(DOWN).shift(3*LEFT)
    randtext = Text("High Frequency Mode").scale(0.5).to_edge(DOWN).shift(3*RIGHT)
    r_label = MathTex(r"r = b - Ax").scale(0.8).shift(4.2*LEFT+1.2*UP)
    self.play(Write(title),Write(r_label),Write(smoothtext),Write(randtext),Write(jacobitext))
    
    it_text = Text("Iteration: 0").scale(0.8)
    self.add(it_text)

    # RNG seed for reproducibility
    np.random.seed(42)
    # number of points
    n = 101
    # x coordinates of line, and the answer
    x_coords = np.linspace(0,1,n)
    u = np.sin(2*math.pi*x_coords)
    A = create1dLaplaceMatrix(n)
    # use the answer to form the RHS
    b = np.matmul(A,u)
    # create an initial guess, but pre-break the pieces by smooth and non-smooth components
    xrand = np.random.rand(n)/4.
    xsmooth = 4.*np.sin(8.*math.pi*x_coords) + 4.*np.sin(4*math.pi*x_coords) + 0.*np.cos(2*math.pi*x_coords)
    x = xrand+xsmooth
    # compute residual
    r = b - np.matmul(A,x)
    
    # setup axes and draw a plot
    ax_main = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-n**2, n**2, 1000], # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},#      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    ).scale(0.5).shift(UP)
    graph_main = ax_main.plot_line_graph(x_values=x_coords, y_values=r, line_color=BLUE, add_vertex_dots=False)

    rsmooth = b - np.matmul(A,xsmooth)
    ax_smooth = ax_main.copy().shift(3.*DOWN+3.5*LEFT)
    graph_smooth = ax_smooth.plot_line_graph(x_values=x_coords, y_values=rsmooth, line_color=BLUE, add_vertex_dots=False)

    rrand = b - np.matmul(A,xrand)
    ax_rand = ax_main.copy().shift(3.*DOWN+3.5*RIGHT)
    graph_rand = ax_rand.plot_line_graph(x_values=x_coords, y_values=rrand, line_color=BLUE, add_vertex_dots=False)

    self.play(Create(ax_main),Create(graph_main))
    self.play(Create(ax_smooth),Create(graph_smooth))
    self.play(Create(ax_rand),Create(graph_rand))
    self.remove(graph_smooth,graph_rand,graph_main)
    
    num_its = 30
    for it in range(0,num_its):
      xsmooth = applyJacobi(xsmooth,b,A,2./3.)
      rsmooth = b - np.matmul(A,xsmooth)
      graph_smooth_new = ax_smooth.plot_line_graph(x_values=x_coords, y_values=rsmooth, line_color=BLUE, add_vertex_dots=False)
      xrand = applyJacobi(xrand,b,A,2./3.)
      rrand = b - np.matmul(A,xrand)
      graph_rand_new = ax_rand.plot_line_graph(x_values=x_coords, y_values=rrand, line_color=BLUE, add_vertex_dots=False)
      x = xrand + xsmooth
      r = b - np.matmul(A,x)
      graph_main_new = ax_main.plot_line_graph(x_values=x_coords, y_values=r, line_color=BLUE, add_vertex_dots=False)

      self.remove(it_text)
      it_text = Text("Iteration: {}".format(it+1)).scale(0.8)
      self.add(it_text)
      
      self.play(ReplacementTransform(graph_smooth,graph_smooth_new),ReplacementTransform(graph_rand,graph_rand_new),ReplacementTransform(graph_main,graph_main_new),run_time=0.2)
      self.remove(graph_smooth_new)
      self.remove(graph_rand_new)
      self.remove(graph_main_new)

    self.add(graph_smooth_new)
    self.add(graph_rand_new)
    self.add(graph_main_new)

    self.wait(3)

# Defines a piecewise linear function from [0,1] to the convex hull of the input points
class PiecewiseLinearFunction:
  def __init__(self,y_values):
    self.y_values = y_values
    self.num_points = len(y_values)
    self.num_elems = self.num_points-1

    # print details
    print("PiecewiseLinearFunction details...")
    print("self.num_points = {}".format(self.num_points))
    print("self.y_values = {}".format(self.y_values))

  def value(self,x):
    # find which interval we're on
    xindex = math.floor(x*(self.num_elems))
    # rescale for the interval
    t = x*(self.num_elems) - xindex
    # interpolate as (1-t)x(i) + t x(i+1)
    return (1-t)*self.y_values[xindex] + t*self.y_values[xindex+1]
    

# A class which can be used to apply a Jacobi smoother for a 1D Laplace problem
# This assumes the Laplacian matrix is [-1, 2, -1]
class JacobiLaplaceSmoother:
  def __init__(self,rhs,omega):
    self.rhs = rhs
    self.num_dofs = len(rhs)
    self.omega = omega
  
  # apply the update x = x + omega*Dinv*(b-Ax) once
  def smooth(self,x):
    out_vector = []

    out_vector.append(x[0] + self.omega*0.5*(self.rhs[0] - (2*x[0]-x[1])))
    for i in range(1,len(x)-1):
      out_vector.append(x[i] + self.omega*0.5*(self.rhs[i] - (2*x[i]-x[i+1]-x[i-1])))
    out_vector.append(x[-1] + self.omega*0.5*(self.rhs[-1] - (2*x[-1]-x[-2])))

    return out_vector

class GridTransfers(Scene):
  def construct(self):
    # num_fine is the number of fine grid points
    # suitable choices include 21, 43, 87
    num_fine = 87
    num_intermediate = int((num_fine-1)/2)
    num_coarse = int((num_intermediate-1)/2)
    
    # setup text labels
    graph1_text = Text("Fine Grid").scale(0.8)
    graph2_text = Text("Intermediate Grid").scale(0.8)
    graph3_text = Text("Coarse Grid").scale(0.8)

    # setup initial axes (will be reused later via copy())
    ax1 = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-1.1, 1.1, 0.2],
      # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},
      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    )
    ax1.scale(0.3).shift(2.5*UP + 2*LEFT)
    
    # draw graph number 1
    np.random.seed(42)
    graph1_text.move_to(ax1).shift(4*RIGHT) # position text to the right of the axes
    graph1_x = np.linspace(0,1,num_fine)
    #graph1_y = 2*np.random.rand(num_fine)-1
    graph1_y = np.sin(10.*math.pi*graph1_x) + np.sin(25.*math.pi*graph1_x) + np.sin(50*math.pi*graph1_x)
    graph1 = ax1.plot_line_graph(x_values=graph1_x, y_values=graph1_y, line_color=BLUE, add_vertex_dots=False) # plot it as a line graph

    # draw graph number 2
    ax2 = ax1.copy().shift(2.5*DOWN)
    graph2_text.move_to(ax2).shift(4*RIGHT)
    graph2_x = np.linspace(0,1,num_intermediate)
    graph2_y = applyLinearInterpR(graph1_y)
    graph2 = ax1.plot_line_graph(x_values=graph2_x, y_values=graph2_y, line_color=RED, add_vertex_dots=False)

    # draw graph number 3
    ax3 = ax2.copy().shift(2.5*DOWN)
    graph3_text.move_to(ax3).shift(4*RIGHT)
    graph3_x = np.linspace(0,1,num_coarse)
    graph3_y = applyLinearInterpR(graph2_y)
    graph3 = ax1.plot_line_graph(x_values=graph3_x, y_values=graph3_y, line_color=GREEN, add_vertex_dots=False)
    
    # animate everything in order
    self.play(Write(graph1_text),Create(ax1))
    self.play(Create(graph1),run_time=3)
    self.wait()
    #self.play(Write(graph2_text),Create(ax2))
    self.play(Create(graph2),run_time=3)
    self.wait()
    #self.play(Write(graph3_text),Create(ax3))
    self.play(Create(graph3),run_time=3)
    self.wait()

class Interpolation(Scene):
  def construct(self):
    # num_points is the main parameter for determining how the plots look
    num_points = 13

    # setup text labels
    graph1_text = MathTex(r"f(x)=\sin(2\pi x)").scale(0.8)
    graph2_text = MathTex(r"f(x)=\sin(4\pi x)").scale(0.8)
    graph3_text = MathTex(r"f(x)=\sin(6\pi x)").scale(0.8)

    # setup initial axes (will be reused later via copy())
    ax1 = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-1.1, 1.1, 0.2],
      # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},
      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    )
    ax1.scale(0.3).shift(2.5*UP + 2*LEFT)

    # draw graph number 1
    #graph = ax1.plot(lambda x: math.sin(2*math.pi*x), color=RED, x_range=[0, 5])
    graph1_text.move_to(ax1).shift(4*RIGHT) # position text to the right of the axes
    graph1_x = np.linspace(0,1,num_points) # create x values
    graph1_y = np.sin(2*math.pi*graph1_x) # create y values
    graph1 = ax1.plot_line_graph(x_values=graph1_x, y_values=graph1_y, line_color=BLUE, add_vertex_dots=False) # plot it as a line graph

    # draw graph number 2
    ax2 = ax1.copy().shift(2.5*DOWN)
    graph2_text.move_to(ax2).shift(4*RIGHT)
    graph2_y = np.sin(4*math.pi*graph1_x)
    graph2 = ax2.plot_line_graph(x_values=graph1_x, y_values=graph2_y, line_color=BLUE, add_vertex_dots=False)

    # draw graph number 3
    ax3 = ax2.copy().shift(2.5*DOWN)
    graph3_text.move_to(ax3).shift(4*RIGHT)
    graph3_y = np.sin(6*math.pi*graph1_x)
    graph3 = ax3.plot_line_graph(x_values=graph1_x, y_values=graph3_y, line_color=BLUE, add_vertex_dots=False)

    # animate everything in order
    self.play(Write(graph1_text),Create(ax1))
    self.play(Create(graph1),run_time=3)
    self.wait()
    self.play(Write(graph2_text),Create(ax2))
    self.play(Create(graph2),run_time=3)
    self.wait()
    self.play(Write(graph3_text),Create(ax3))
    self.play(Create(graph3),run_time=3)
    self.wait()

class JacobiSmoother(Scene):
  def construct(self):
    # num_fine is the main parameter for this scene
    num_fine = 89
    h = 1./(num_fine-1)
    np.random.seed(42)
    graph_x = np.linspace(0,1,num_fine)
    graph_y = 2*np.random.rand(num_fine)-1
    
    sol = np.sin(2*math.pi*graph_x)
    A = create1dLaplaceMatrix(num_fine)

    #b = np.matmul(A,sol)
    b = -((2*math.pi)**2)*np.sin(2*math.pi*graph_x)
    b[0] = 0
    b[num_fine-1] = 0

    # text
    #top_text = Text('Jacobi Smoother').shift(3*UP)
    smoother_text = MathTex(r"x^{(n+1)} = x^{(n)} + \omega D^{-1} (b - Ax^{(n)})").shift(3*UP)

    # setup axes and graph
    ax = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-1.1, 1.1, 0.2],
      # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},
      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    )
    graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
    graphs = [graph]

    # apply smoother
    new_y = applyJacobi(graph_y,b,A,0.5)
    graphs.append(ax.plot_line_graph(x_values=graph_x, y_values=new_y, line_color=BLUE, add_vertex_dots=False))

    # animate it!
    it_text = Text("Iteration: 0").shift(3*DOWN)
    self.add(smoother_text, ax, graph, it_text)
    self.wait()
    self.remove(it_text)
    it_text = Text("Iteration: 1").shift(3*DOWN)
    self.add(it_text)
    self.play(Transform(graphs[0],graphs[1]))
    for it in range(0,100):
      graph_y = new_y
      new_y = applyJacobi(graph_y,b,A,0.6)
      graphs.append(ax.plot_line_graph(x_values=graph_x, y_values=new_y, line_color=BLUE, add_vertex_dots=False))
      self.remove(graphs[it])
      self.remove(it_text)
      it_text = Text("Iteration: {}".format(it+1)).shift(3*DOWN)
      self.add(it_text)
      self.play(Transform(graphs[it+1],graphs[it+2]),run_time=0.2)

    # # generate random values, throw them in a vector, then turn it into a plottable function
    # initial_vector_function = PiecewiseLinearFunction(initial_vector)

    # # create the axes
    # ax = Axes(x_range=[0, 1, 0.001], y_range=[0, 1])

    # # plot the initial vector
    # graph = ax.plot(lambda x: initial_vector_function.value(x), color=RED, x_range=[0, 0.999], use_smoothing=False)

    # # smooth the initial vector
    # smoother = JacobiLaplaceSmoother(rhs_vector,0.6)
    # smoothed_vector = smoother.smooth(initial_vector)
    # smoothed_vector_function = PiecewiseLinearFunction(smoothed_vector)

    # # plot the smoothed vector
    # graph_smoothed = ax.plot(lambda x: smoothed_vector_function.value(x), color=RED, x_range=[0, 0.999], use_smoothing=False)

    # # create dots based on the graph
    # moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
    # dot_1 = Dot(ax.i2gp(graph.t_min, graph))
    # dot_2 = Dot(ax.i2gp(graph.t_max, graph))

    # #self.add(ax, graph, dot_1, dot_2, moving_dot)
    # self.add(ax, graph, top_text)
    # self.play(Transform(graph, graph_smoothed))

class HeatEquation(Scene):
  def construct(self):
    # Generic setup
    title_text = Text("Jacobi's Method — Clamped Heat Equation").scale(0.8).shift(3*UP)
    it_text = Text("Iteration: 0").scale(0.5).shift(1.8*UP)
    eq_text = MathTex(r"\mathbf{x} = \mathbf{x} + \omega D^{-1}(\mathbf{b}-A\mathbf{x})").scale(0.8).shift(2.4*UP)

    # Physics setup
    # heat equation with left and right ends clamped, no forcing
    bc_left = 0.4
    bc_right = 1.0

    # Mesh/discretization setup
    n = 101
    h = 1./(n-1)
    graph_x = np.linspace(0,1,n)
    graph_y = graph_x*0.
    graph_y[0] = bc_left
    graph_y[-1] = bc_right
    n_text = Text(f"n = {n} variables").scale(0.5).shift(2*DOWN)
    self.play(Write(title_text),Write(it_text),Write(eq_text),Write(n_text))
    
    # Matrix and RHS setup
    num_its = 101
    A = create1dLaplaceMatrix(n)
    b = graph_y
    sol = np.linalg.solve(A,b)    
    
    # Setup axes
    ax = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-1.1, 1.1, 0.2],
      # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},
      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    ).shift(1.5*DOWN+0.5*RIGHT)
    graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
    graph_sol = ax.plot_line_graph(x_values=graph_x, y_values=sol, line_color=RED, add_vertex_dots=False)["line_graph"]
    graph_sol = DashedVMobject(graph_sol)
    sol_text = Text("true solution",color=RED).scale(0.4).shift(2.*LEFT+UP)
    numsol_text = Text("numerical solution",color=BLUE).scale(0.4).shift(DOWN+LEFT)
    box = SurroundingRectangle(graph, buff=0.05, color=WHITE)
    bcno_text = Text("0°C").scale(0.5).align_to(box,[-1., -1., 0.]).shift(0.8*LEFT)
    bclo_text = Text("40°C").scale(0.5).align_to(box,[-1., 0., 0.]).shift(LEFT+0.4*DOWN)
    bchi_text = Text("100°C").scale(0.5).align_to(box,[1., 1., 0.]).shift(1.2*RIGHT)
    self.play(Create(graph),Create(numsol_text),Create(box),Create(bcno_text),Create(bclo_text),Create(bchi_text))
    self.wait(1)
    self.play(Create(graph_sol),Create(sol_text))
    self.wait(1)
    
    # Animate Jacobi
    # first 10 iterations are done one at a time
    for it in range(1,min(10,num_its)):
      graph_y = applyJacobi(graph_y,b,A,2./3.)
      new_graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
      self.remove(it_text)
      it_text = Text(f"Iteration: {it}").scale(0.5).shift(1.8*UP)
      self.add(it_text)
      self.remove(graph)
      self.play(ReplacementTransform(graph,new_graph),run_time=0.3)
      graph = new_graph
      self.remove(new_graph)

    # now step 5 at a time
    for it in range(min(10,num_its),num_its,5):
      for subit in range(0,5):
        graph_y = applyJacobi(graph_y,b,A,2./3.)
      new_graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
      self.remove(it_text)
      it_text = Text(f"Iteration: {it}").scale(0.5).shift(1.8*UP)
      self.add(it_text)
      self.remove(graph)
      self.play(ReplacementTransform(graph,new_graph),run_time=0.3)
      graph = new_graph
      self.remove(new_graph)
    self.add(new_graph)
    conc_text = Text("Jacobi's method only transmits information locally!",color=BLUE).scale(0.6).shift(3*DOWN)
    self.play(Create(conc_text))
    self.wait(3)

class HeatEquationMultigrid(Scene):
  def construct(self):
    # Generic setup
    title_text = Text("Jacobi's Method — Clamped Heat Equation").scale(0.8).shift(3*UP)
    it_text = Text("Iteration: 0").scale(0.5).shift(1.8*UP)
    eq_text = MathTex(r"\mathbf{x} = \mathbf{x} + \omega D^{-1}(\mathbf{b}-A\mathbf{x})").scale(0.8).shift(2.4*UP)

    # Physics setup
    # heat equation with left and right ends clamped, no forcing
    bc_left = 0.4
    bc_right = 1.0

    # Mesh/discretization setup
    n = 101
    h = 1./(n-1)
    graph_x = np.linspace(0,1,n)
    graph_y = graph_x*0.
    graph_y[0] = bc_left
    graph_y[-1] = bc_right
    n_text = Text(f"n = {n} variables").scale(0.5).shift(2*DOWN)
    self.play(Write(title_text),Write(it_text),Write(eq_text),Write(n_text))
    
    # Matrix and RHS setup
    num_its = 5
    A = create1dLaplaceMatrix(n)
    b = graph_y
    sol = np.linalg.solve(A,b)    
    
    # Setup axes
    ax = Axes(
      x_range=[0, 1.1, 0.2],
      y_range=[-1.1, 1.1, 0.2],
      # note: last label in the range won't show up if using np.arange(0,5,1)
      x_axis_config={"numbers_to_include": np.arange(0, 1.1, 1)},
      y_axis_config={"numbers_to_include": np.arange(-1, 1.1, 0.4)},
      tips=False,
    ).shift(1.5*DOWN+0.5*RIGHT)
    graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
    graph_sol = ax.plot_line_graph(x_values=graph_x, y_values=sol, line_color=RED, add_vertex_dots=False)["line_graph"]
    graph_sol = DashedVMobject(graph_sol)
    sol_text = Text("true solution",color=RED).scale(0.4).shift(2.*LEFT+UP)
    numsol_text = Text("numerical solution",color=BLUE).scale(0.4).shift(DOWN+LEFT)
    box = SurroundingRectangle(graph, buff=0.05, color=WHITE)
    bcno_text = Text("0°C").scale(0.5).align_to(box,[-1., -1., 0.]).shift(0.8*LEFT)
    bclo_text = Text("40°C").scale(0.5).align_to(box,[-1., 0., 0.]).shift(LEFT+0.4*DOWN)
    bchi_text = Text("100°C").scale(0.5).align_to(box,[1., 1., 0.]).shift(1.2*RIGHT)
    self.play(Create(graph),Create(numsol_text),Create(box),Create(bcno_text),Create(bclo_text),Create(bchi_text))
    self.wait(1)
    self.play(Create(graph_sol),Create(sol_text))
    self.wait(1)
    
    # Animate multigrid
    # first 10 iterations are done one at a time
    for it in range(1,num_its):
      graph_y = applyJacobi(graph_y,b,A,2./3.)
      new_graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
      self.remove(it_text)
      it_text = Text(f"Iteration: {it}").scale(0.5).shift(1.8*UP)
      self.add(it_text)
      self.remove(graph)
      self.play(ReplacementTransform(graph,new_graph),run_time=0.3)

      # coarse grid
      coarse_y = applyLinearInterpR(graph_y)
      coarse_x = np.linspace(0,1,len(coarse_y))
      coarse_b = 0.*coarse_y
      coarse_b[0] = bc_left
      coarse_b[-1] = bc_right
      coarse_graph = ax.plot_line_graph(x_values=coarse_x, y_values=coarse_y, line_color=GREEN, add_vertex_dots=False)
      self.remove(new_graph)
      self.play(ReplacementTransform(new_graph,coarse_graph))
      
      coarse_A = create1dLaplaceMatrix(len(coarse_y))
      coarse_Ainv = np.linalg.inv(coarse_A)
      coarse_y = np.matmul(coarse_Ainv,coarse_b)
      coarse_graph_sol = ax.plot_line_graph(x_values=coarse_x, y_values=coarse_y, line_color=GREEN, add_vertex_dots=False)
      self.remove(coarse_graph)
      self.play(ReplacementTransform(coarse_graph,coarse_graph_sol))

      fine_y = applyLinearInterpP(coarse_y)
      graph_y = fine_y
      graph = ax.plot_line_graph(x_values=graph_x, y_values=graph_y, line_color=BLUE, add_vertex_dots=False)
      self.remove(coarse_graph_sol)
      self.play(ReplacementTransform(coarse_graph_sol,graph))

    self.add(new_graph)
    conc_text = Text("Multigrid transmits information globally!",color=BLUE).scale(0.6).shift(3*DOWN)
    self.play(Create(conc_text))
    self.wait(3)
