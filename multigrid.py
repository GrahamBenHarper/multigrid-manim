from manim import *
import numpy as np
import math
import random

def createSineVector(coordinates: np.array):
  return np.sin(coordinates)

def createRandomVector(coordinates: np.array):
  return np.random.rand(len(coordinates))

# apply the smoother x = x + omega*Dinv(b-Ax)
def applySmoother(x: np.array, b: np.array, A: np.array, omega: float):
  Dinv = np.diag(1./np.diag(A))
  r = b - np.matmul(A,x)
  return x + omega*np.matmul(Dinv,r)

# apply a linear interpolation restrictor matrix-free
# Briggs uses a factor of 1/4 for R
def applyLinearInterpR(x: np.array):
  if(len(x)%2==0):
    print("Attempting to apply LinearInterpP on a vector with {} entries! Only odd numbers of entries are supported!".format(len(x)))
  
  scale = 4. #2.*math.sqrt(2.)
  xout = np.zeros(int((len(x)-1)/2))
  for i in range(0,len(xout)):
    xout[i] = (x[2*i] + 2.*x[2*i+1] + x[2*i+2])/scale
  
  return xout

# apply a linear interpolation prolongator matrix-free
# Briggs uses a factor of 1/2 for P
def applyLinearInterpP(x: np.array):  
  xout = np.zeros(2*len(x)+1)

  scale = 2. #2.*math.sqrt(2.)
  xout[0] = x[0]/scale
  for i in range(0,len(x)):
    xout[2*i+1] = 2*x[i]/scale
    xout[2*i+2] = (x[i]+x[i+1])/scale
  
  return xout

# defines a piecewise linear function from [0,1] to the convex hull of the input points
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
    A = np.diag(2*np.ones(num_fine)) + np.diag(-np.ones(num_fine-1),1) + np.diag(-np.ones(num_fine-1),-1)
    A = A/(h**2)
    #b = np.matmul(A,sol)
    b = -((2*math.pi)**2)*np.sin(2*math.pi*graph_x)

    # fix boundary conditions
    A[0,:] = np.zeros(num_fine)
    A[0,0] = 1
    A[num_fine-1,:] = np.zeros(num_fine)
    A[num_fine-1,num_fine-1] = 1
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
    new_y = applySmoother(graph_y,b,A,0.5)
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
      new_y = applySmoother(graph_y,b,A,0.6)
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
