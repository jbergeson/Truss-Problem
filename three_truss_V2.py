import numpy as np 
from truss import truss
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver


import numpy as np 
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
        self.options.declare("n_known", desc = "Number of known forces acting on node")
        self.options.declare("n_unknown", desc = "Number of unknown trusses acting on node")
        self.options.declare("solve_first", default = "x", desc = "Tells component which direction to sum the forces in first")

    def setup(self):
        
        self.add_output("new_truss 1", units = "N", desc = "Force in first unknown truss")
        self.add_input("new_direction 1", units = "rad", desc = "Direction of first unknown force acting on node")
        self.add_input("known_force 1", units = "N", desc = "First known force acting on node")
        self.add_input("old_direction 1", units = "rad", desc = "Direction of first known force acting on node")

        if (self.options["n_unknown"] > 1):
            self.add_output("new_truss 2", units = "N", desc = "Force in second unknown truss")
            self.add_input("new_direction 2", units = "rad", desc = "Direction of second unknown force acting on node")
        
        if (self.options["n_known"] > 1):
            self.add_input("known_force 2", units = "N", desc = "Second known force acting on node")
            self.add_input("old_direction 2", units = "rad", desc = "Direction of second known force acting on node")

#What should I take for derivatives?
        self.declare_partials("new_truss*", "known_force*", method = "fd")
        self.declare_partials("new_truss*", "old_direction*", method = "fd")
        self.declare_partials("new_truss*", "new_direction*", method = "fd")

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


class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("x_reaction_n1", F * (3 ** .5) / 2, units = "N", desc = "Horizontal reaction force of pinnned joint on node 1")
        indeps.add_output("y_reaction_n1", F, units = "N", desc = "Vertical reaction force of pinnned joint on node 1")
        indeps.add_output("x_reaction_direction_n1", math.pi, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 1")
        indeps.add_output("y_reaction_direction_n1", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 1")
        indeps.add_output("reaction_n2", F * (3 ** .5) / 2, units = "N", desc = "Reaction force of roller joint on node 2")
        indeps.add_output("reaction_direction_n2", 0, units = "rad", desc = "Direction of reaction force of roller joint on node 2")
        indeps.add_output("truss1_n1", math.pi * 11 / 6, units = "rad", desc = "Direction of truss1 at node 1")
        indeps.add_output("truss2_n1", math.pi * 3 / 2, units = "rad", desc = "Direction of truss2 at node 1")
        indeps.add_output("truss2_n2", math.pi / 2, units = "rad", desc = "Direction of truss2 at node 2")
        indeps.add_output("truss3_n2", math.pi / 6, units = "rad", desc = "Direction of truss2 at node 2")
        
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("L")
        # What about the units for the subsystem?
        
        self.add_subsystem("node1", Node(n_known = 2, n_unknown = 2))
        self.add_subsystem("node2", Node(n_known = 2, n_unknown = 1))
        truss1 = self.add_subsystem("truss1", truss())
        truss2 = self.add_subsystem("truss2", truss())
        truss3 = self.add_subsystem("truss3", truss())

        
        self.connect("indeps.x_reaction_n1", "node1.known_force 1")
        self.connect("indeps.y_reaction_n1", "node1.known_force 2")
        self.connect("indeps.x_reaction_direction_n1", "node1.old_direction 1")
        self.connect("indeps.y_reaction_direction_n1", "node1.old_direction 2")
        self.connect("indeps.truss1_n1", "node1.new_direction 1")
        self.connect("indeps.truss2_n1", "node1.new_direction 2")
        
        self.connect("indeps.reaction_n2", "node2.known_force 1")
        self.connect("indeps.reaction_direction_n2", "node2.old_direction 1")
        self.connect("node1.new_truss 2", "node2.known_force 2")
        self.connect("indeps.truss2_n2", "node2.old_direction 2")
        self.connect("indeps.truss3_n2", "node2.new_direction 1")

        self.connect("node1.new_truss 1", ["truss1.P"])
        self.connect("node1.new_truss 2", ["truss2.P"])
        self.connect("node2.new_truss 1", ["truss3.P"])
        
        self.add_subsystem("obj_cmp", ExecComp("obj = L * (A1 + A2 + A3)", A1 = 0.0, A2 = 0.0, A3 = 0.0, L = 1.0))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con3", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L", ["obj_cmp.L"])
        self.connect("truss1.sigma", ["con1.sigma"])
        self.connect("truss2.sigma", ["con2.sigma"])
        self.connect("truss3.sigma", ["con3.sigma"])
        self.connect("indeps.A1", ["truss1.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["truss2.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["truss3.A", "obj_cmp.A3"])

if __name__ == "__main__":

    prob = Problem()
    prob.model = Truss_Analysis()

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    # prob.driver.options["tol"] = 1e-8

    prob.model.add_design_var("indeps.A1", lower = .001, upper = 100)
    prob.model.add_design_var("indeps.A2", lower = .001, upper = 100)
    prob.model.add_design_var("indeps.A3", lower = .001, upper = 100)
    prob.model.add_objective("obj_cmp.obj")
    prob.model.add_constraint("con1.con", lower = 0)
    prob.model.add_constraint("con2.con", lower = 0)
    prob.model.add_constraint("con3.con", lower = 0)

    prob.setup()
    # prob.check_partials(compact_print = True)
    prob.set_solver_print(level = 0)

    prob.model.approx_totals()

    prob.run_driver()

    print("minimum found at")
    print("A1 = ", prob["indeps.A1"])
    print("truss1.P", prob["truss1.P"])
    print("A2 = ", prob["indeps.A2"])
    print("truss2.P", prob["truss2.P"])
    print("A3 = ", prob["indeps.A3"])
    print("truss3.P = ", prob["truss3.P"])




