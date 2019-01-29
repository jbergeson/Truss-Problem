import numpy as np 
from truss import truss
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Truss_Analysis(Group):
    
    def setup(self):
        
        ##F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("P1", 0)
        indeps.add_output("P2", F)
        indeps.add_output("P3", F)
        indeps.add_output("P4", F)
        indeps.add_output("P5", -F * (2 ** .5))
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("A4")
        indeps.add_output("A5")
        indeps.add_output("L1")
        indeps.add_output("L2")
        # What about the units for the subsystem?
        trussAB = self.add_subsystem("trussAB", truss())
        trussBC = self.add_subsystem("trussBC", truss())
        trussCD = self.add_subsystem("trussCD", truss())
        trussDE = self.add_subsystem("trussDA", truss())
        trussEA = self.add_subsystem("trussAC", truss())

        self.connect("indeps.P1", ["trussAB.P"])
        self.connect("indeps.P2", ["trussBC.P"])
        self.connect("indeps.P3", ["trussCD.P"])
        self.connect("indeps.P4", ["trussDA.P"])
        self.connect("indeps.P5", ["trussAC.P"])
        
        self.add_subsystem("obj_cmp", ExecComp("obj = L1 * (A1 + A2 + A3 + A4) + L2 * A5", A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0, A5 = 0.0, L1 = 1.0, L2 = 2 ** .5))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con3", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con4", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con5", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L1", ["obj_cmp.L1"])
        self.connect("indeps.L2", ["obj_cmp.L2"])
        self.connect("trussAB.sigma", ["con1.sigma"])
        self.connect("trussBC.sigma", ["con2.sigma"])
        self.connect("trussCD.sigma", ["con3.sigma"])
        self.connect("trussDA.sigma", ["con4.sigma"])
        self.connect("trussAC.sigma", ["con5.sigma"])
        self.connect("indeps.A1", ["trussAB.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["trussBC.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["trussCD.A", "obj_cmp.A3"])
        self.connect("indeps.A4", ["trussDA.A", "obj_cmp.A4"])
        self.connect("indeps.A5", ["trussAC.A", "obj_cmp.A5"])


prob = Problem()
prob.model = Truss_Analysis()

prob.driver = ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
# prob.driver.options["tol"] = 1e-8

prob.model.add_design_var("indeps.A1", lower = 0.0001, upper = 100)
prob.model.add_design_var("indeps.A2", lower = 0.0001, upper = 100)
prob.model.add_design_var("indeps.A3", lower = 0.0001, upper = 100)
prob.model.add_design_var("indeps.A4", lower = 0.0001, upper = 100)
prob.model.add_design_var("indeps.A5", lower = 0.0001, upper = 100)
prob.model.add_objective("obj_cmp.obj")
prob.model.add_constraint("con1.con", lower = 0)
prob.model.add_constraint("con2.con", lower = 0)
prob.model.add_constraint("con3.con", lower = 0)
prob.model.add_constraint("con4.con", lower = 0)
prob.model.add_constraint("con5.con", lower = 0)

prob.setup()
# prob.check_partials(compact_print = True)
prob.set_solver_print(level = 0)

prob.model.approx_totals()

prob.run_driver()

print("minimum found at")
print("A1 = ", prob["indeps.A1"])
print("P1 = ", prob["indeps.P1"])
print("A2 = ", prob["indeps.A2"])
print("P2 = ", prob["indeps.P2"])
print("A3 = ", prob["indeps.A3"])
print("P3 = ", prob["indeps.P3"])
print("A4 = ", prob["indeps.A4"])
print("P4 = ", prob["indeps.P4"])
print("A5 = ", prob["indeps.A5"])
print("P5 = ", prob["indeps.P5"])

