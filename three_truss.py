import numpy as np 
from truss import truss
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4 * 10 ** 7
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("P1", F / 2)
        indeps.add_output("P2", -F)
        indeps.add_output("P3", F)
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("A3")
        indeps.add_output("L")
        # What about the units for the subsystem?
        trussAB = self.add_subsystem("trussAB", truss())
        trussBC = self.add_subsystem("trussBC", truss())
        trussCA = self.add_subsystem("trussCA", truss())

        self.connect("indeps.P1", ["trussAB.P"])
        self.connect("indeps.P2", ["trussBC.P"])
        self.connect("indeps.P3", ["trussCA.P"])
        
        self.add_subsystem("obj_cmp", ExecComp("obj = L * (A1 + A2 + A3)", A1 = 0.0, A2 = 0.0, A3 = 0.0, L = 1.0))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con3", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L", ["obj_cmp.L"])
        self.connect("trussAB.sigma", ["con1.sigma"])
        self.connect("trussBC.sigma", ["con2.sigma"])
        self.connect("trussCA.sigma", ["con3.sigma"])
        self.connect("indeps.A1", ["trussAB.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["trussBC.A", "obj_cmp.A2"])
        self.connect("indeps.A3", ["trussCA.A", "obj_cmp.A3"])


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
print("P1 = ", prob["indeps.P1"])
print("A2 = ", prob["indeps.A2"])
print("P2 = ", prob["indeps.P2"])
print("A3 = ", prob["indeps.A3"])
print("P3 = ", prob["indeps.P3"])




