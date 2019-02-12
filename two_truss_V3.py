import math
import numpy as np 
from truss_V3 import truss, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver, ArmijoGoldsteinLS

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("n0_x_reaction", math.pi, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 0")
        indeps.add_output("n0_y_reaction", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 0")
        indeps.add_output("n1_x_reaction", 0, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 1")
        indeps.add_output("n1_y_reaction", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 1")
        indeps.add_output("n0_truss0", math.pi * 11 / 6, units = "rad", desc = "Direction of truss 0 at node 0")
        indeps.add_output("n1_truss1", math.pi / 6, units = "rad", desc = "Direction of truss 1 at node 1")
        indeps.add_output("n2_truss1", math.pi * 7 / 6, units = "rad", desc = "Direction of truss 1 at node 2")
        indeps.add_output("n2_truss0", math.pi * 5 / 6, units = "rad", desc = "Direction of truss 0 at node 2")
        indeps.add_output("ext", F, units = "N", desc = "Force applied to truss structure")
        indeps.add_output("ext_direction", math.pi * 3 / 2, units = "rad", desc = "Direction of force applied to truss structure")

        indeps.add_output("A0")
        indeps.add_output("A1")
        indeps.add_output("L1")
        
        cycle = self.add_subsystem("cycle", Group())
        cycle.add_subsystem("node0", Node(n_forces_out = 2, n_forces_in = 1))
        cycle.add_subsystem("node1", Node(n_forces_out = 2, n_forces_in = 1))
        cycle.add_subsystem("node2", Node(n_forces_out = 2, n_forces_in = 0, n_external_forces = 1))
        self.add_subsystem("truss0", truss())
        self.add_subsystem("truss1", truss())

        #Node 0 connections
        self.connect("indeps.n0_x_reaction", "cycle.node0.direction0_out")
        self.connect("indeps.n0_y_reaction", "cycle.node0.direction1_out")
        self.connect("indeps.n0_truss0", "cycle.node0.direction0_in")
        #Node 1 connections
        self.connect("indeps.n1_x_reaction", "cycle.node1.direction0_out")
        self.connect("indeps.n1_y_reaction", "cycle.node1.direction1_out")
        self.connect("indeps.n1_truss1", "cycle.node1.direction0_in")
        #Node 2 connections
        self.connect("indeps.n2_truss1", "cycle.node2.direction0_out")
        self.connect("indeps.n2_truss0", "cycle.node2.direction1_out")
        self.connect("indeps.ext_direction", "cycle.node2.direction0_external")
        self.connect("indeps.ext", "cycle.node2.force0_external")
        #Inter-node connections
        self.connect("cycle.node2.force0_out", ["cycle.node1.force0_in", "truss1.P"])
        self.connect("cycle.node2.force1_out", ["cycle.node0.force0_in", "truss0.P"])
        
        cycle.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.options["iprint"] = 2
        cycle.linear_solver = DirectSolver()


        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A0 + A1)", A0 = 0.0, A1 = 0.0, L1 = 1.0))
        self.add_subsystem("con0", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("truss0.sigma", ["con0.sigma"])
        self.connect("truss1.sigma", ["con1.sigma"])
        self.connect("indeps.A0", ["truss0.A", "obj_cmp.A0"])
        self.connect("indeps.A1", ["truss1.A", "obj_cmp.A1"])

if __name__ == "__main__":

    prob = Problem()
    prob.model = Truss_Analysis()

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    # prob.driver.options["tol"] = 1e-8

    prob.model.add_design_var("indeps.A0", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A1", lower = 0.001, upper = 100)
    prob.model.add_objective("obj_cmp.obj")
    prob.model.add_constraint("con0.con", lower = 0)
    prob.model.add_constraint("con1.con", lower = 0)

    prob.setup()
    prob.check_partials(compact_print = True)
    prob.set_solver_print(level = 0)

    prob.model.approx_totals()

    prob.run_driver()

    print("minimum found at")
    print("A0 = ", prob["indeps.A0"])
    print("truss0.P", prob["truss0.P"])
    print("A1 = ", prob["indeps.A1"])
    print("truss1.P", prob["truss1.P"])
    print("n0_x_reaction", prob["indeps.n0_x_reaction"])
    print("n0_y_reaction", prob["indeps.n0_y_reaction"])
    print("n1_x_reaction", prob["indeps.n1_x_reaction"])
    print("n1_y_reaction", prob["indeps.n1_y_reaction"])