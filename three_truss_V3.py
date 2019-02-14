import math
import numpy as np 
from truss_V3 import Beam, Node
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver, ArmijoGoldsteinLS

class Truss_Analysis(Group):
    
    def setup(self):
        
        F = 4 * 10 ** 7
        
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("n0_x_reaction_direction", 0, units = "rad", desc = "Direction of horizontal reaction force of pinned joint on node 0")
        indeps.add_output("n0_y_reaction_direction", math.pi / 2, units = "rad", desc = "Direction of vertical reaction force of pinned joint on node 0")
        indeps.add_output("n1_x_reaction_direction", 0, units = "rad", desc = "Direction of reaction force of roller joint on node 1")
        indeps.add_output("n0_beam0", math.pi * 11 / 6, units = "rad", desc = "Direction of beam 0 at node 0")
        indeps.add_output("n0_beam1", math.pi * 3 / 2, units = "rad", desc = "Direction of beam 1 at node 0")
        indeps.add_output("n1_beam1", math.pi / 2, units = "rad", desc = "Direction of beam 1 at node 1")
        indeps.add_output("n1_beam2", math.pi / 6, units = "rad", desc = "Direction of beam 1 at node 1")
        indeps.add_output("n2_beam0", math.pi * 5 / 6, units = "rad", desc = "Direction of beam 0 at node 2")
        indeps.add_output("n2_beam2", math.pi * 7 / 6, units = "rad", desc = "Direction of beam 2 at node 2")
        indeps.add_output("ext", F, units = "N", desc = "Force applied to beam structure")
        indeps.add_output("ext_direction", math.pi * 3 / 2, units = "rad", desc = "Direction of force applied to beam structure")

        indeps.add_output("A0")
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("L1")
        
        cycle = self.add_subsystem("cycle", Group())
        cycle.add_subsystem("node0", Node(n_loads = 2, n_reactions = 2))
        cycle.add_subsystem("node1", Node(n_loads = 2, n_reactions = 1))
        cycle.add_subsystem("node2", Node(n_loads = 2, n_external_forces = 1))
        cycle.add_subsystem("beam0", Beam())
        cycle.add_subsystem("beam1", Beam())
        cycle.add_subsystem("beam2", Beam())
    
       
        #Node 0 connections
        self.connect("indeps.n0_x_reaction_direction", "cycle.node0.direction0_reaction")
        self.connect("indeps.n0_y_reaction_direction", "cycle.node0.direction1_reaction")
        self.connect("indeps.n0_beam0", "cycle.node0.direction0_load")
        self.connect("indeps.n0_beam1", "cycle.node0.direction1_load")
        #Node 1 connections
        self.connect("indeps.n1_x_reaction_direction", "cycle.node1.direction0_reaction")
        self.connect("indeps.n1_beam1", "cycle.node1.direction0_load")
        self.connect("indeps.n1_beam2", "cycle.node1.direction1_load")
        #Node 2 connections
        self.connect("indeps.n2_beam0", "cycle.node2.direction0_load")
        self.connect("indeps.n2_beam2", "cycle.node2.direction1_load")
        self.connect("indeps.ext_direction", "cycle.node2.direction0_ext")
        self.connect("indeps.ext", "cycle.node2.force0_ext")
        #Inter-node connections
        self.connect("cycle.node0.load_out0", "cycle.beam0.force0")
        self.connect("cycle.node0.load_out1", "cycle.beam1.force0")
        self.connect("cycle.node1.load_out0", "cycle.beam1.force1")
        self.connect("cycle.node1.load_out1", "cycle.beam2.force0")
        self.connect("cycle.node2.load_out0", "cycle.beam0.force1")
        self.connect("cycle.node2.load_out1", "cycle.beam2.force1")
        self.connect("cycle.beam0.beam_force", ["cycle.node0.load_in0", "cycle.node2.load_in0"])
        self.connect("cycle.beam1.beam_force", ["cycle.node0.load_in1", "cycle.node1.load_in0"])
        self.connect("cycle.beam2.beam_force", ["cycle.node1.load_in1", "cycle.node2.load_in1"])
        
        cycle.nonlinear_solver = NewtonSolver()
        cycle.nonlinear_solver.options['atol'] = 1e-7
        cycle.nonlinear_solver.options['solve_subsystems'] = True
        cycle.nonlinear_solver.options["iprint"] = 2
        cycle.linear_solver = DirectSolver()


        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A0 + A1 + A2 + A3 + A4)"))
        self.add_subsystem("con0", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("cycle.beam0.sigma", ["con0.sigma"])
        self.connect("cycle.beam1.sigma", ["con1.sigma"])
        self.connect("cycle.beam2.sigma", ["con2.sigma"])
        self.connect("indeps.A0", ["cycle.beam0.A", "obj_cmp.A0"])
        self.connect("indeps.A1", ["cycle.beam1.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["cycle.beam2.A", "obj_cmp.A2"])

if __name__ == "__main__":

    prob = Problem()
    prob.model = Truss_Analysis()

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("indeps.A0", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A1", lower = 0.001, upper = 100)
    prob.model.add_design_var("indeps.A2", lower = 0.001, upper = 100)
    prob.model.add_objective("obj_cmp.obj")
    prob.model.add_constraint("con0.con", lower = 0)
    prob.model.add_constraint("con1.con", lower = 0)
    prob.model.add_constraint("con2.con", lower = 0)

    prob.setup(force_alloc_complex = True)
    prob.check_partials(compact_print = True, method = "cs")
    # prob.check_totals()
    # exit()
    prob.run_driver()
    # prob.run_model()



    print("minimum found at")
    print("A0 = ", prob["indeps.A0"])
    print("beam0.beam_force", prob["cycle.beam0.beam_force"])
    print("A1 = ", prob["indeps.A1"])
    print("beam1.beam_force", prob["cycle.beam1.beam_force"])
    print("A2 = ", prob["indeps.A2"])
    print("beam2.beam_force", prob["cycle.beam2.beam_force"])
    print("n0_x_reaction", prob["cycle.node0.reaction0"])
    print("n0_y_reaction", prob["cycle.node0.reaction1"])
    print("n1_x_reaction", prob["cycle.node1.reaction0"])