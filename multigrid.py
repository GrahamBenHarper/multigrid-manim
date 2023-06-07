from manim import *
import math
import random

# defines a random function from [0,1] to [0,1]
class RandomFunction:
  def __init__(self,num_points):
    self.num_points = num_points
    self.num_elems = num_points-1

    # generate random values
    random.seed(42)
    random_values = []
    for i in range(0,num_points):
      random_values.append(random.uniform(0,1))
    self.random_values = random_values

    # print details
    print("RandomFunction details...")
    print("self.num_points = {}".format(self.num_points))
    print("self.random_values = {}".format(self.random_values))

  def value(self,x):
    # find which interval we're on
    xindex = math.floor(x*(self.num_elems))
    # rescale for the interval
    t = x*(self.num_elems) - xindex

    print(xindex)
    
    return (1-t)*self.random_values[xindex] + t*self.random_values[xindex+1]

class FollowingGraphCamera(MovingCameraScene):
  def construct(self):
    self.camera.frame.save_state()

    rand_function = RandomFunction(51)

    # create the axes and the curve
    ax = Axes(x_range=[0, 1, 0.001], y_range=[0, 1])
    graph = ax.plot(lambda x: rand_function.value(x), color=BLUE, x_range=[0, 0.999], use_smoothing=False)

    # create dots based on the graph
    moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
    dot_1 = Dot(ax.i2gp(graph.t_min, graph))
    dot_2 = Dot(ax.i2gp(graph.t_max, graph))

    self.add(ax, graph, dot_1, dot_2, moving_dot)
