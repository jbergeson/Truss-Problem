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
        self.options("n_forces")
        # self.options.declare("n_known", desc = "Number of known forces acting on node")
        # self.options.declare("n_unknown", desc = "Number of unknown trusses acting on node")
        # self.options.declare("solve_first", default = "x", desc = "Tells component which direction to sum the forces in first")

    def setup(self):
        
        for n in range(self.options["m_forces"]):
            n_force = "force{}".format(n - 1)
            n_direction = "direction{}".format(n - 1)
            self.add_input(n_direction, units = "rad", desc = "Direction of nth force on node")
            self.add_output(n_force, units = "N", desc = "Magnitude of nth force on node")

        self.declare_partials("x", "force*")
        self.declare_partials("x", "direction*")
        self.declare_partials("y", "force*")
        self.declare_partials("y", "direction*")
        
        # self.add_output("new_truss 1", units = "N", desc = "Force in first unknown truss")
        # self.add_input("new_direction 1", units = "rad", desc = "Direction of first unknown force acting on node")
        # self.add_input("known_force 1", units = "N", desc = "First known force acting on node")
        # self.add_input("old_direction 1", units = "rad", desc = "Direction of first known force acting on node")

        # if (self.options["n_unknown"] > 1):
        #     self.add_output("new_truss 2", units = "N", desc = "Force in second unknown truss")
        #     self.add_input("new_direction 2", units = "rad", desc = "Direction of second unknown force acting on node")
        
        # if (self.options["n_known"] > 1):
        #     self.add_input("known_force 2", units = "N", desc = "Second known force acting on node")
        #     self.add_input("old_direction 2", units = "rad", desc = "Direction of second known force acting on node")

#What should I take for derivatives?
        self.declare_partials("new_truss*", "known_force*", method = "fd")
        self.declare_partials("new_truss*", "old_direction*", method = "fd")
        self.declare_partials("new_truss*", "new_direction 1", method = "fd")

    def compute(self, inputs, outputs):
        forces = [0]
        forces[0] = inputs["known_force 1"]
        old_directions = [0]
        old_directions[0] = inputs["old_direction 1"]
        new_directions = [0]
        new_directions[0] = inputs["new_direction 1"]
        if (self.options["n_unknown"] > 1):
            new_directions.append(inputs["new_direction 2"])
        if (self.options["n_known"] > 1):
            forces.append(inputs["known_force 2"])
            old_directions.append(inputs["old_direction 2"])



        if (self.options["solve_first"] == "x"):
            x_sum = 0
            for i in range(self.options["n_known"]):
                x_sum += forces[i] * math.cos(old_directions[i])
            outputs["new_truss 1"] = -x_sum / math.cos(new_directions[0])
            if (self.options["n_unknown"] > 1):
                y_sum = 0
                for i in range(self.options["n_known"]):
                    y_sum += forces[i] * math.sin(old_directions[i])
                y_sum += outputs["new_truss 1"] * math.sin(new_directions[0])
                outputs["new_truss 2"] = -y_sum / math.sin(new_directions[1])

        else:
            y_sum = 0
            for i in range(self.options["n_known"]):
                y_sum += forces[i] * math.sin(old_directions[i])
            outputs["new_truss 1"] = -y_sum / math.sin(new_directions[0])
            if (self.options["n_unknown"] > 1):
                x_sum = 0
                for i in range(self.options["n_known"]):
                    x_sum += forces[i] * math.cos(old_directions[i])
                x_sum += outputs["new_truss 1"] * math.cos(new_directions[0])
                outputs["new_truss 2"] = -x_sum / math.cos(new_directions[1])