import numpy as np 
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