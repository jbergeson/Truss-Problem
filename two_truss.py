import numpy as np 
from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class truss(ExplicitComponent):

    def setup(self):

        self.add_input("A", val = 0.)
        self.add_input("P", val = 0.)

        self.add_output("sigma", val = 0.)

        self.declare_partials("sigma", "A")
        self.declare_partials("sigma", "P")
    
    def compute(self, inputs, outputs):
        
        A = inputs["A"]
        P = inputs["P"]

        outputs["sigma"] = P / (A * 1000000)
    
    def compute_partials(self, inputs, J):

        A = inputs["A"]
        P = inputs["P"]

        J["sigma", "A"] = -P / (A ** 2)
        if (P > 0):
            J["sigma", "P"] = 1 / A
        else:
            J["sigma", "P"] = -1 / A

class Truss_Analysis(Group):
    
    def setup(self):
        
        #F is in N, A is in m^2, L is in m, and sigma is in MPa
        F = 4000000
        indeps = self.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("P1", -F)
        indeps.add_output("P2", F)
        indeps.add_output("A1")
        indeps.add_output("A2")
        indeps.add_output("L")

        trussAB = self.add_subsystem("trussAB", truss())
        trussBC = self.add_subsystem("trussBC", truss())

        self.connect("indeps.P1", ["trussAB.P"])
        self.connect("indeps.P2", ["trussBC.P"])
        
        self.add_subsystem("obj_cmp", ExecComp("obj = L * (A1 + A2)", A1 = 1.0, A2 = 1.0, L = 1.0))
        self.add_subsystem("con1", ExecComp("con = 400 - abs(sigma)"))
        self.add_subsystem("con2", ExecComp("con = 400 - abs(sigma)"))

        self.connect("indeps.L", ["obj_cmp.L"])
        self.connect("trussAB.sigma", ["con1.sigma"])
        self.connect("trussBC.sigma", ["con2.sigma"])
        self.connect("indeps.A1", ["trussAB.A", "obj_cmp.A1"])
        self.connect("indeps.A2", ["trussBC.A", "obj_cmp.A2"])


prob = Problem()
prob.model = Truss_Analysis()

prob.driver = ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
# prob.driver.options["tol"] = 1e-8

prob.model.add_design_var("indeps.A1", lower = .001, upper = 100)
prob.model.add_design_var("indeps.A2", lower = .001, upper = 100)
prob.model.add_objective("obj_cmp.obj")
prob.model.add_constraint("con1.con", lower = 0)
prob.model.add_constraint("con2.con", lower = 0)

prob.setup()
prob.check_partials(compact_print = True)
prob.set_solver_print(level = 0)

prob.model.approx_totals()

prob.run_driver()

print("minimum found at")
print(prob["indeps.A1"])
print(prob["indeps.P1"])
print(prob["indeps.A2"])
print(prob["indeps.P2"])
print(prob["con1.con"])
print(prob["con2.con"])



