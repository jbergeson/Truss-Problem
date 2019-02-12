import math
import numpy as np 
from truss_V3 import Beam, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver, ArmijoGoldsteinLS

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("n0_x_reaction_direction", math.pi, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 0")
        indeps.add_output("n0_y_reaction_direction", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 0")
        indeps.add_output("n1_x_reaction_direction", 0, units = "rad", desc = "Direction of reaction force of roller joint on node 1")
        indeps.add_output("n0_beam0", 0, units = "rad", desc = "Direction of beam 0 at node 0")
        indeps.add_output("n0_beam1", math.pi * 3 / 2, units = "rad", desc = "Direction of beam 1 at node 0")
        indeps.add_output("n1_beam1", math.pi / 2, units = "rad", desc = "Direction of beam 1 at node 1")
        indeps.add_output("n1_beam2", 0, units = "rad", desc = "Direction of beam 2 at node 1")
        indeps.add_output("n1_beam3", math.pi / 4, units = "rad", desc = "Direction of beam 3 at node 1")
        indeps.add_output("n2_beam2", math.pi, units = "rad", desc = "Direction of beam 2 at node 2")
        indeps.add_output("n2_beam4", math.pi / 2, units = "rad", desc = "Direction of beam 4 at node 2")
        indeps.add_output("n3_beam0", math.pi, units = "rad", desc = "Direction of beam 0 at node 3")
        indeps.add_output("n3_beam3", math.pi * 5 / 4, units = "rad", desc = "Direction of beam 3 at node 3")
        indeps.add_output("n3_beam4", math.pi * 3 / 2, units = "rad", desc = "Direction of beam 4 at node 3")
        indeps.add_output("ext", F, units = "N", desc = "Force applied to beam structure")
        indeps.add_output("ext_direction", math.pi * 3 / 2, units = "rad", desc = "Direction of force applied to beam structure")

        indeps.add_output("A0")
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("A4")
        indeps.add_output("L1")
        
        cycle = self.add_subsystem("cycle", Group())
        cycle.add_subsystem("node0", Node(n_loads = 2, n_reactions = 2))
        cycle.add_subsystem("node1", Node(n_loads = 3, n_reactions = 1))
        cycle.add_subsystem("node2", Node(n_loads = 2, n_external_forces = 1))
        cycle.add_subsystem("node3", Node(n_loads = 3))
        cycle.add_subsystem("beam0", Beam())
        cycle.add_subsystem("beam1", Beam())
        cycle.add_subsystem("beam2", Beam())
        cycle.add_subsystem("beam3", Beam())
        cycle.add_subsystem("beam4", Beam())
    
       
        #Node 0 connections
        self.connect("indeps.n0_x_reaction_direction", "cycle.node0.direction0")
        self.connect("indeps.n0_y_reaction_direction", "cycle.node0.direction1")
        self.connect("indeps.n0_beam0", "cycle.node0.direction2")
        self.connect("indeps.n0_beam1", "cycle.node0.direction3")
        #Node 1 connections
        self.connect("indeps.n1_x_reaction_direction", "cycle.node1.direction0")
        self.connect("indeps.n1_beam1", "cycle.node1.direction1")
        self.connect("indeps.n1_beam2", "cycle.node1.direction2")
        self.connect("indeps.n1_beam3", "cycle.node1.direction3")
        #Node 2 connections
        self.connect("indeps.n2_beam2", "cycle.node2.direction0")
        self.connect("indeps.n2_beam4", "cycle.node2.direction1")
        self.connect("indeps.ext_direction", "cycle.node2.direction0_ext")
        self.connect("indeps.ext", "cycle.node2.force0_ext")
        #Node 3 connections
        self.connect("indeps.n3_beam0", "cycle.node3.direction0")
        self.connect("indeps.n3_beam3", "cycle.node3.direction1")
        self.connect("indeps.n3_beam4", "cycle.node3.direction2")
        #Inter-node connections
        self.connect("cycle.node0.load2_out", "cycle.beam0.force0")
        self.connect("cycle.node0.load3_out", "cycle.beam1.force0")
        self.connect("cycle.node1.load1_out", "cycle.beam1.force1")
        self.connect("cycle.node1.load2_out", "cycle.beam2.force0")
        self.connect("cycle.node1.load3_out", "cycle.beam3.force0")
        self.connect("cycle.node2.load0_out", "cycle.beam2.force1")
        self.connect("cycle.node2.load1_out", "cycle.beam4.force0")
        self.connect("cycle.node3.load0_out", "cycle.beam0.force1")
        self.connect("cycle.node3.load1_out", "cycle.beam3.force1")
        self.connect("cycle.node3.load2_out", "cycle.beam4.force1")
        self.connect("cycle.beam0.beam_force", ["cycle.node0.load2_in", "cycle.node3.load0_in"])
        self.connect("cycle.beam1.beam_force", ["cycle.node0.load3_in", "cycle.node1.load1_in"])
        self.connect("cycle.beam2.beam_force", ["cycle.node1.load2_in", "cycle.node2.load0_in"])
        self.connect("cycle.beam3.beam_force", ["cycle.node1.load3_in", "cycle.node3.load1_in"])
        self.connect("cycle.beam4.beam_force", ["cycle.node2.load1_in", "cycle.node3.load2_in"])
        
        cycle.nonlinear_solver = NewtonSolver()
        cycle.nonlinear_solver.options['atol'] = 1e-7
        cycle.nonlinear_solver.options["iprint"] = 2
        cycle.linear_solver = DirectSolver()


        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A0 + A1 + A2 + A3 + A4)"))
        self.add_subsystem("con0", ExecComp("con = 400000000 - abs(P / A)"))
        self.add_subsystem("con1", ExecComp("con = 400000000 - abs(P / A)"))
        self.add_subsystem("con2", ExecComp("con = 400000000 - abs(P / A)"))
        self.add_subsystem("con3", ExecComp("con = 400000000 - abs(P / A)"))
        self.add_subsystem("con4", ExecComp("con = 400000000 - abs(P / A)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("cycle.beam0.beam_force", ["con0.P"])
        self.connect("cycle.beam1.beam_force", ["con1.P"])
        self.connect("cycle.beam2.beam_force", ["con2.P"])
        self.connect("cycle.beam3.beam_force", ["con3.P"])
        self.connect("cycle.beam4.beam_force", ["con4.P"])
        self.connect("indeps.A0", ["con0.A", "obj_cmp.A0"])
        self.connect("indeps.A1", ["con1.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["con2.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["con3.A", "obj_cmp.A3"])
        self.connect("indeps.A4", ["con4.A", "obj_cmp.A4"])

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
    # prob.check_partials(compact_print = False)
    # prob.check_totals()
    # exit()
    prob.run_driver()
    # prob.run_model()



    print("minimum found at")
    print("A0 = ", prob["indeps.A0"])
    print("beam0.P", prob["cycle.beam0.beam_force"])
    print("A1 = ", prob["indeps.A1"])
    print("beam1.P", prob["cycle.beam1.beam_force"])
    print("A2 = ", prob["indeps.A2"])
    print("beam2.P", prob["cycle.beam2.beam_force"])
    print("A3 = ", prob["indeps.A3"])
    print("beam3.P", prob["cycle.beam3.beam_force"])
    print("A4 = ", prob["indeps.A4"])
    print("beam4.P", prob["cycle.beam4.beam_force"])
    print("n0_x_reaction", prob["cycle.node0.load0_out"])
    print("n0_y_reaction", prob["cycle.node0.load1_out"])
    print("n1_x_reaction", prob["cycle.node1.load1_out"])