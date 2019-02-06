import math
import numpy as np 
from truss_V3 import truss, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("n0_x_reaction", math.pi, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 0")
        indeps.add_output("n0_y_reaction", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 0")
        indeps.add_output("n1_x_reaction", 0, units = "rad", desc = "Direction of reaction force of roller joint on node 1")
        indeps.add_output("n0_truss0", 0, units = "rad", desc = "Direction of truss 0 at node 0")
        indeps.add_output("n0_truss1", math.pi * 3 / 2, units = "rad", desc = "Direction of truss 1 at node 0")
        indeps.add_output("n1_truss1", math.pi / 2, units = "rad", desc = "Direction of truss 1 at node 1")
        indeps.add_output("n1_truss2", 0, units = "rad", desc = "Direction of truss 2 at node 1")
        indeps.add_output("n1_truss3", math.pi / 4, units = "rad", desc = "Direction of truss 3 at node 1")
        indeps.add_output("n2_truss2", math.pi, units = "rad", desc = "Direction of truss 2 at node 2")
        indeps.add_output("n2_truss4", math.pi / 2, units = "rad", desc = "Direction of truss 4 at node 2")
        indeps.add_output("n3_truss0", math.pi, units = "rad", desc = "Direction of truss 0 at node 3")
        indeps.add_output("n3_truss3", math.pi * 5 / 4, units = "rad", desc = "Direction of truss 3 at node 3")
        indeps.add_output("n3_truss4", math.pi * 3 / 2, units = "rad", desc = "Direction of truss 4 at node 3")
        indeps.add_output("ext", F, units = "N", desc = "Force applied to truss structure")
        indeps.add_output("ext_direction", math.pi * 3 / 2, units = "rad", desc = "Direction of force applied to truss structure")

        indeps.add_output("A0")
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("A4")
        indeps.add_output("L1")
        
        # self.add_subsystem("node0", Node(n_forces = 4, n_external = 0))
        # self.add_subsystem("node1", Node(n_forces = 4, n_external = 0))
        # self.add_subsystem("node2", Node(n_forces = 2, n_external = 1))
        # self.add_subsystem("node3", Node(n_forces = 3, n_external = 0))
        cycle = self.add_subsystem("cycle", Group())
        cycle.add_subsystem("node0", Node(n_forces = 4, n_external = 0))
        cycle.add_subsystem("node1", Node(n_forces = 4, n_external = 0))
        cycle.add_subsystem("node2", Node(n_forces = 2, n_external = 1))
        cycle.add_subsystem("node3", Node(n_forces = 3, n_external = 0))
        self.add_subsystem("truss0", truss())
        self.add_subsystem("truss1", truss())
        self.add_subsystem("truss2", truss())
        self.add_subsystem("truss3", truss())
        self.add_subsystem("truss4", truss())

        #Node 0 connections
        self.connect("indeps.n0_x_reaction", "cycle.node0.direction0")
        self.connect("indeps.n0_y_reaction", "cycle.node0.direction1")
        self.connect("indeps.n0_truss0", "cycle.node0.direction2")
        self.connect("indeps.n0_truss1", "cycle.node0.direction3")
        #Node 1 connections
        self.connect("indeps.n1_x_reaction", "cycle.node1.direction0")
        self.connect("indeps.n1_truss1", "cycle.node1.direction1")
        self.connect("indeps.n1_truss2", "cycle.node1.direction2")
        self.connect("indeps.n1_truss3", "cycle.node1.direction3")
        #Node 2 connections
        self.connect("indeps.n2_truss2", "cycle.node2.direction0")
        self.connect("indeps.n2_truss4", "cycle.node2.direction1")
        self.connect("indeps.ext_direction", "cycle.node2.external_direction0")
        self.connect("indeps.ext", "cycle.node2.external_force0")
        #Node 3 connections
        self.connect("indeps.n3_truss0", "cycle.node3.direction0")
        self.connect("indeps.n3_truss3", "cycle.node3.direction1")
        self.connect("indeps.n3_truss4", "cycle.node3.direction2")
        #Inter-node connections
        self.connect("cycle.node0.force3", ["cycle.node1.force1", "truss1.P"])
        self.connect("cycle.node0.force2", ["cycle.node3.force0", "truss0.P"])
        self.connect("cycle.node1.force2", ["cycle.node2.force0", "truss2.P"])
        self.connect("cycle.node1.force3", ["cycle.node3.force1", "truss3.P"])
        self.connect("cycle.node2.force1", ["cycle.node3.force2", "truss4.P"])
        
        cycle.nonlinear_solver = NonlinearBlockGS()

        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A0 + A1 + A2 + A3 + A4)", A0 = 0.0, A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0, L1 = 1.0))
        self.add_subsystem("con0", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con3", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con4", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("truss0.sigma", ["con0.sigma"])
        self.connect("truss1.sigma", ["con1.sigma"])
        self.connect("truss2.sigma", ["con2.sigma"])
        self.connect("truss3.sigma", ["con3.sigma"])
        self.connect("truss4.sigma", ["con4.sigma"])
        self.connect("indeps.A0", ["truss0.A", "obj_cmp.A0"])
        self.connect("indeps.A1", ["truss1.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["truss2.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["truss3.A", "obj_cmp.A3"])
        self.connect("indeps.A4", ["truss4.A", "obj_cmp.A4"])

if __name__ == "__main__":

    prob = Problem()
    prob.model = Truss_Analysis()

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    # prob.driver.options["tol"] = 1e-8

    prob.model.add_design_var("indeps.A0", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A1", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A2", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A3", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A4", lower = 0.001, upper = 100)
    prob.model.add_objective("obj_cmp.obj")
    prob.model.add_constraint("con0.con", lower = 0)
    prob.model.add_constraint("con1.con", lower = 0)
    prob.model.add_constraint("con2.con", lower = 0)
    prob.model.add_constraint("con3.con", lower = 0)
    prob.model.add_constraint("con4.con", lower = 0)

    prob.setup()
    # prob.check_partials(compact_print = True)
    prob.set_solver_print(level = 0)

    prob.model.approx_totals()

    prob.run_driver()

    print("minimum found at")
    print("A0 = ", prob["indeps.A0"])
    print("truss0.P", prob["truss0.P"])
    print("A1 = ", prob["indeps.A1"])
    print("truss1.P", prob["truss1.P"])
    print("A2 = ", prob["indeps.A2"])
    print("truss2.P", prob["truss2.P"])
    print("A3 = ", prob["indeps.A3"])
    print("truss3.P", prob["truss3.P"])
    print("A4 = ", prob["indeps.A4"])
    print("truss4.P", prob["truss4.P"])
    print("n0_x_reaction", prob["indeps.n0_x_reaction"])
    print("n0_y_reaction", prob["indeps.n0_y_reaction"])
    print("n1_x_reaction", prob["indeps.n1_x_reaction"])