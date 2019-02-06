import math
import numpy as np 
from truss_V2 import truss, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("x_reaction_n1", F * ((3 ** .5) + 2) / 2, units = "N", desc = "Horizontal reaction force of pinnned joint on node 1")
        indeps.add_output("y_reaction_n1", F, units = "N", desc = "Vertical reaction force of pinned joint on node 1")
        indeps.add_output("x_reaction_direction_n1", math.pi, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 1")
        indeps.add_output("y_reaction_direction_n1", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 1")
        indeps.add_output("reaction_n2", F * ((3 ** .5) + 2) / 2, units = "N", desc = "Horizontal reaction force of pinnned joint on node 2")
        indeps.add_output("reaction_direction_n2", 0, units = "rad", desc = "Direction of reaction force of roller joint on node 2")
        indeps.add_output("truss1_n1", 0, units = "rad", desc = "Direction of truss1 at node1")
        indeps.add_output("truss2_n1", math.pi * 3 / 2, units = "rad", desc = "Direction of truss2 at node1")
        indeps.add_output("truss2_n2", math.pi / 2, units = "rad", desc = "Direction of truss2 at node2")
        indeps.add_output("truss3_n2", math.pi / 4, units = "rad", desc = "Direction of truss3 at node2")
        indeps.add_output("truss4_n2", 0, units = "rad", desc = "Direction of truss4 at node2")
        indeps.add_output("truss4_n3", math.pi, units = "rad", desc = "Direction of truss4 at node3")
        indeps.add_output("truss5_n3", math.pi / 6, units = "rad", desc = "Direction of truss5 at node3")
        indeps.add_output("truss6_n3", math.pi / 2, units = "rad", desc = "Direction of truss6 at node3")
        indeps.add_output("truss5_n4", math.pi * 7 / 6, units = "rad", desc = "Direction of truss5 at node4")
        indeps.add_output("truss7_n4", math.pi * 5 / 6, units = "rad", desc = "Direction of truss7 at node4")
        indeps.add_output("F", F, units = "N", desc = "Force applied to truss structure")
        indeps.add_output("F_direction", math.pi * 3 / 2, units = "rad", desc = "Direction of force applied to truss structure")

        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("A4")
        indeps.add_output("A5")
        indeps.add_output("A6")
        indeps.add_output("A7")
        indeps.add_output("L1")
        indeps.add_output("L2")
        # What about the units for the subsystem?
        
        self.add_subsystem("node1", Node(n_known = 2, n_unknown = 2))
        self.add_subsystem("node2", Node(n_known = 2, n_unknown = 2, solve_first = "y"))
        self.add_subsystem("node3", Node(n_known = 1, n_unknown = 2))
        self.add_subsystem("node4", Node(n_known = 2, n_unknown = 1))
        self.add_subsystem("truss1", truss())
        self.add_subsystem("truss2", truss())
        self.add_subsystem("truss3", truss())
        self.add_subsystem("truss4", truss())
        self.add_subsystem("truss5", truss())
        self.add_subsystem("truss6", truss())
        self.add_subsystem("truss7", truss())

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
        self.connect("indeps.truss4_n2", "node2.new_direction 2")

        self.connect("node2.new_truss 2", "node3.known_force 1")
        self.connect("indeps.truss4_n3", "node3.old_direction 1")
        self.connect("indeps.truss5_n3", "node3.new_direction 1")
        self.connect("indeps.truss6_n3", "node3.new_direction 2")

        self.connect("node3.new_truss 1", "node4.known_force 1")
        self.connect("indeps.truss5_n4", "node4.old_direction 1")
        self.connect("indeps.F", "node4.known_force 2")
        self.connect("indeps.F_direction", "node4.old_direction 2")
        self.connect("indeps.truss7_n4", "node4.new_direction 1")
        
        self.connect("node1.new_truss 1", ["truss1.P"])
        self.connect("node1.new_truss 2", ["truss2.P"])
        self.connect("node2.new_truss 1", ["truss3.P"])
        self.connect("node2.new_truss 2", ["truss4.P"])
        self.connect("node3.new_truss 1", ["truss5.P"])
        self.connect("node3.new_truss 2", ["truss6.P"])
        self.connect("node4.new_truss 1", ["truss7.P"])
        
        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A1 + A2 + A3 + A4 + A5 + A7) + L2 * A6", A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0, A5 = 0.0, A6 = 0.0, A7 = 0.0, L1 = 1.0, L2 = 2 ** .5))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con3", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con4", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con5", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con6", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con7", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("indeps.L2", ["obj_cmp.L2"])
        self.connect("truss1.sigma", ["con1.sigma"])
        self.connect("truss2.sigma", ["con2.sigma"])
        self.connect("truss3.sigma", ["con3.sigma"])
        self.connect("truss4.sigma", ["con4.sigma"])
        self.connect("truss5.sigma", ["con5.sigma"])
        self.connect("truss6.sigma", ["con6.sigma"])
        self.connect("truss7.sigma", ["con7.sigma"])
        self.connect("indeps.A1", ["truss1.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["truss2.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["truss3.A", "obj_cmp.A3"])
        self.connect("indeps.A4", ["truss4.A", "obj_cmp.A4"])
        self.connect("indeps.A5", ["truss5.A", "obj_cmp.A5"])
        self.connect("indeps.A6", ["truss6.A", "obj_cmp.A6"])
        self.connect("indeps.A7", ["truss7.A", "obj_cmp.A7"])

if __name__ == "__main__":

    prob = Problem()
    prob.model = Truss_Analysis()

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    # prob.driver.options["tol"] = 1e-8

    prob.model.add_design_var("indeps.A1", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A2", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A3", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A4", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A5", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A6", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A7", lower = 0.001, upper = 100)
    prob.model.add_objective("obj_cmp.obj")
    prob.model.add_constraint("con1.con", lower = 0)
    prob.model.add_constraint("con2.con", lower = 0)
    prob.model.add_constraint("con3.con", lower = 0)
    prob.model.add_constraint("con4.con", lower = 0)
    prob.model.add_constraint("con5.con", lower = 0)
    prob.model.add_constraint("con6.con", lower = 0)
    prob.model.add_constraint("con7.con", lower = 0)

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
    print("truss3.P", prob["truss3.P"])
    print("A4 = ", prob["indeps.A4"])
    print("truss4.P", prob["truss4.P"])
    print("A5 = ", prob["indeps.A5"])
    print("truss5.P", prob["truss5.P"])
    print("A6 = ", prob["indeps.A6"])
    print("truss6.P", prob["truss6.P"])
    print("A7 = ", prob["indeps.A7"])
    print("truss7.P", prob["truss7.P"])



