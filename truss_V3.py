import numpy as np 
from openmdao.api import ExplicitComponent, ImplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Truss(ImplicitComponent):

    # def setup(self):

        # self.add_input("A", val = 0., units = "m**2", desc = "Cross sectional area of member")
        # self.add_input("P", val = 0., units = "N", desc = "Tensile/Compressive force in member")

        # self.add_output("sigma", val = 0., units = "MPa", desc = "Stress in member")

        # self.declare_partials("sigma", "A")
        # self.declare_partials("sigma", "P")

    # def compute(self, inputs, outputs):
        
    #     A = inputs["A"]
    #     P = inputs["P"]

    #     outputs["sigma"] = P / (A * 1000000)
    
    # def compute_partials(self, inputs, J):

    #     A = inputs["A"]
    #     P = inputs["P"]

    #     J["sigma", "A"] = -P / (1000000 * A ** 2)
    #     J["sigma", "P"] = 1 / (A * 1000000)
    def setup(self):
        
        self.add_input("force0", val = 1., units = "N", desc = "Force on 0th end of beam")
        self.add_input("force1", val = 1., units = "N", desc = "Force on 1st end of beam")
        self.add_output("beam_force", val = 1., units = "N", desc = "Force in the beam")
        self.declare_partials("beam_force", "force0")
        self.declare_partials("beam_force", "force1")

    def apply_nonlinear(self, inputs, output, residuals):
        residuals["beam_force"] = inputs["force0"] - inputs["force1"]

    def linearize(self, inputs, outputs, partials):
        partials["beam_force", "force0"] = 1
        partials["beam_force", "force1"] = -1


class Node(ImplicitComponent):
 
    def initialize(self):
        self.options.declare("n_loads", default = 2, desc = "Number of loads on node")
        self.options.declare("n_external_forces", default = 0, desc = "Number of external forces on node")

    def setup(self):

        for n in range(self.options["n_loads"]):

            n_load_in = "load{}_in".format(n)
            n_direction = "direction{}".format(n)
            n_load_out = "load{}_out".format(n)
            self.add_input(n_load_in, units = "N", desc = "Input load on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of load on node")
            self.add_output(n_load_out, units = "N", desc = "Output load on node")

        for m in range(self.options["n_external_forces"]):
            m_force = "force{}_ext".format(m)
            m_direction = "direction{}_ext".format(m)
            self.add_input(m_force, units = "N", desc = "External force on node")
            self.add_input(m_direction, units = "rad", desc = "Direction of external force on node")

        for i in range(self.options["n_loads"]):
            n_load_out = "load{}_out".format(i)
            self.declare_partials(n_load_out, "load*")
            self.declare_partials(n_load_out, "direction*")
            if (self.options["n_external_forces"] > 0):

                self.declare_partials(n_load_out, "force*")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["load0_out"] = 0
        residuals["load1_out"] = 0
        for n in range(self.options["n_loads"]):
            
            load = "load{}_in".format(n)
            direction = "direction{}".format(n)
            residuals["load0_out"] += inputs[load] * np.cos(inputs[direction])
            residuals["load1_out"] += inputs[load] * np.sin(inputs[direction])
        
        for m in range(self.options["n_external_forces"]):
            force = "force{}_ext".format(m)
            direction = "direction{}_ext".format(m)
            residuals["load0_out"] += inputs[force] * np.cos(inputs[direction])
            residuals["load1_out"] += inputs[force] * np.sin(inputs[direction])

        i = 2
        while (i < self.options["n_loads"]):
            
            load_in = "load{}_in".format(i)
            load_out = "load{}_out".format(i)
            residuals[load_out] = outputs[load_out] - inputs[load_in]
            i += 1

    def linearize(self, inputs, outputs, partials):
        
        for n in range(self.options["n_loads"]):

            load = "load{}_in".format(n)
            direction = "direction{}".format(n)
            partials["load0_out", load] = np.cos(inputs[direction])
            partials["load0_out", direction] = -inputs[load] * np.sin(inputs[direction])
            partials["load1_out", load] = np.sin(inputs[direction])
            partials["load1_out", direction] = inputs[load] * np.cos(inputs[direction])

        for m in range(self.options["n_external_forces"]):
            force = "force{}_ext".format(m)
            direction = "direction{}_ext".format(m)
            partials["load0_out", force] = np.cos(inputs[direction])
            partials["load0_out", direction] = -inputs[force] * np.sin(inputs[direction])
            partials["load1_out", force] = np.sin(inputs[direction])
            partials["load1_out", direction] = inputs[force] * np.cos(inputs[direction])
        
        i = 2
        while (i < self.options["n_loads"]):
            load_out = "load{}_out".format(i)
            load_in = "load{}_in".format(i)
            direction = "direction{}".format(i)
            partials[load_out, load_out] = -inputs[load_in]
            partials[load_out, direction] = outputs[load_out]
            i += 1