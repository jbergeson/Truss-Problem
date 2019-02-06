import numpy as np 
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver
import math
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class truss(ExplicitComponent):

    def setup(self):

        self.add_input("A", val = 0., units = "m**2", desc = "Cross sectional area of member")
        self.add_input("P", val = 0., units = "N", desc = "Tensile/Compressive force in member")

        self.add_output("sigma", val = 0., units = "MPa", desc = "Stress in member")

        self.declare_partials("sigma", "A")
        self.declare_partials("sigma", "P")
    
    def compute(self, inputs, outputs):
        
        A = inputs["A"]
        P = inputs["P"]

        outputs["sigma"] = P / (A * 1000000)
    
    def compute_partials(self, inputs, J):

        A = inputs["A"]
        P = inputs["P"]

        J["sigma", "A"] = -P / (1000000 * A ** 2)
        J["sigma", "P"] = 1 / (A * 1000000)

class Node(ExplicitComponent):
    
    def initialize(self):
        self.options.declare("n_forces")
        self.options.declare("n_external")

    def setup(self):
        
        for n in range(self.options["n_forces"]):
            n_force = "force{}".format(n)
            n_direction = "direction{}".format(n)
            self.add_input(n_direction, units = "rad", desc = "Direction of nth force on node")
            self.add_output(n_force, units = "N", desc = "Magnitude of nth force on node")

        for n in range(self.options["n_external"]):
            force = "external_force{}".format(n)
            direction = "external_direction{}".format(n)
            self.add_input(force, units = "N", desc = "Magnitude of external force applied to node")
            self.add_input(direction, units = "rad", desc = "Direction of external force applied to node")

        self.declare_partials("x", "force*")
        self.declare_partials("x", "direction*")
        self.declare_partials("y", "force*")
        self.declare_partials("y", "direction*")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["x"] = 0
        residuals["y"] = 0
        for n in range(self.options["n_forces"]):
            residuals["x"] += outputs["force{}".format(n)] * math.cos(inputs["direction{}".format(n)])
            residuals["y"] += outputs["force{}".format(n)] * math.sin(inputs["direction{}".format(n)])
        
        for m in range(self.options["n_external"]):
            residuals["x"] += inputs["external_force{}".format(m)] * math.cos(inputs["external_direction{}".format(m)])
            residuals["y"] += inputs["external_force{}".formal(m)] * math.sin(inputs["external_direction{}".format(m)])
            
    def linearize(inputs, outputs, partials):
        for n in range(self.options["n_forces"]):
            n_force = "force{}".format(n)
            n_direction = "direction{}".format(n)
            partials["x", n_force] = math.cos(inputs[n_direction])
            partials["x", n_direction] = -outputs[n_force] * math.sin(inputs[n_direction])
            partials["y", n_force] = math.sin(inputs[n_direction])
            partials["y", n_direction] = outputs[n_force] * math.cos(inputs[n_direction])