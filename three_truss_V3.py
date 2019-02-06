import numpy as np 
from truss_V2 import truss, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver
import numpy as np 
import math
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

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
        self.add_subsystem("truss1", truss())
        self.add_subsystem("truss2", truss())
        self.add_subsystem("truss3", truss())

        
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
    prob.check_partials(compact_print = True)
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



