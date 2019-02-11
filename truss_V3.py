import numpy as np 
from openmdao.api import ExplicitComponent, ImplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

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

class Node(ImplicitComponent):
 
    def initialize(self):
        self.options.declare("n_forces_out", default = 2, desc = "Number of forces on this node which are not the output of another node, cannot exceed 2")
        self.options.declare("n_forces_in", default = 1, desc = "Number of forces on this node which are the output of another node")
        self.options.declare("n_external_forces", default = 0)

    def setup(self):
        
        for n in range(self.options["n_forces_out"]):
            n_force = "force{}_out".format(n)
            n_direction = "direction{}_out".format(n)
            self.add_input(n_direction, units = "rad", desc = "Direction of nth force out of node")
            self.add_output(n_force, units = "N", desc = "Magnitude of nth force out of node")

        for n in range(self.options["n_forces_in"]):
            n_force = "force{}_in".format(n)
            n_direction = "direction{}_in".format(n)
            self.add_input(n_direction, units = "rad", desc = "Direction of nth force into node")
            self.add_input(n_force, units = "N", desc = "Magnitude of nth force into node")

        for n in range(self.options["n_external_forces"]):
            force = "force{}_external".format(n)
            direction = "direction{}_external".format(n)
            self.add_input(force, units = "N", desc = "Magnitude of external force applied to node")
            self.add_input(direction, units = "rad", desc = "Direction of external force applied to node")
        
        for n in range(self.options["n_forces_out"]):
            n_force = "force{}_out".format(n)
            self.declare_partials(n_force, "direction*")
            self.declare_partials(n_force, "force*")

    def apply_nonlinear(self, inputs, outputs, residuals):
        
        residuals["force0_out"] = 0
        for i in range(self.options["n_forces_out"]):
            residuals["force0_out"] += outputs["force{}_out".format(i)] * np.cos(inputs["direction{}_out".format(i)])
        for n in range(self.options["n_forces_in"]):
            residuals["force0_out"] += inputs["force{}_in".format(n)] * np.cos(inputs["direction{}_in".format(n)])
        for m in range(self.options["n_external_forces"]):
            residuals["force0_out"] += inputs["force{}_external".format(m)] * np.cos(inputs["direction{}_external".format(m)])
        
        residuals["force1_out"] = 0
        for i in range(self.options["n_forces_out"]):
            residuals["force1_out"] += outputs["force{}_out".format(i)] * np.sin(inputs["direction{}_out".format(i)])
        for n in range(self.options["n_forces_in"]):
            residuals["force0_out"] += inputs["force{}_in".format(n)] * np.sin(inputs["direction{}_in".format(n)])
        for m in range(self.options["n_external_forces"]):
            residuals["force0_out"] += inputs["force{}_external".format(m)] * np.sin(inputs["direction{}_external".format(m)])

    def linearize(self, inputs, outputs, partials):
        
        for i in range(self.options["n_forces_out"]):
            n_force = "force{}_out".format(i)
            n_direction = "direction{}_out".format(i)
            partials["force0_out", n_force] =  np.cos(inputs[n_direction])
            partials["force0_out", n_direction] =  -outputs[n_force] * np.sin(inputs[n_direction])

        for n in range(self.options["n_forces_in"]):
            n_force = "force{}_in".format(n)
            n_direction = "direction{}_in".format(n)
            partials["force0_out", n_force] =  np.cos(inputs[n_direction])
            partials["force0_out", n_direction] =  -inputs[n_force] * np.sin(inputs[n_direction])

        for m in range(self.options["n_external_forces"]):
            n_force = "force{}_external".format(m)
            n_direction = "direction{}_external".format(m)
            partials["force0_out", n_force] =  np.cos(inputs[n_direction])
            partials["force0_out", n_direction] =  -inputs[n_force] * np.sin(inputs[n_direction])

        for i in range(self.options["n_forces_out"]):
            n_force = "force{}_out".format(i)
            n_direction = "direction{}_out".format(i)
            partials["force1_out", n_force] =  np.sin(inputs[n_direction])
            partials["force1_out", n_direction] =  outputs[n_force] * np.cos(inputs[n_direction])

        for n in range(self.options["n_forces_in"]):
            n_force = "force{}_in".format(n)
            n_direction = "direction{}_in".format(n)
            partials["force1_out", n_force] =  np.sin(inputs[n_direction])
            partials["force1_out", n_direction] =  inputs[n_force] * np.cos(inputs[n_direction])

        for m in range(self.options["n_external_forces"]):
            n_force = "force{}_external".format(m)
            n_direction = "direction{}_external".format(m)
            partials["force1_out", n_force] =  np.sin(inputs[n_direction])
            partials["force1_out", n_direction] =  inputs[n_force] * np.cos(inputs[n_direction])