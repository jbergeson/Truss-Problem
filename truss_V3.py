import numpy as np 
from openmdao.api import ExplicitComponent, ImplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Beam(ImplicitComponent):

    def setup(self):
        
        self.add_input("force0", val = 1., units = "N", desc = "Force on 0th end of beam")
        self.add_input("force1", val = 1., units = "N", desc = "Force on 1st end of beam")
        self.add_input("A", val = 1., units = "m**2", desc = "Cross sectional area of the beam")
        self.add_output("beam_force", val = 1., units = "N", desc = "Force in the beam")

        self.add_output('sigma', val=1, units='MPa')

        self.declare_partials("beam_force", "force*")
        self.declare_partials("sigma", "beam_force")
        self.declare_partials("sigma", "sigma")
        self.declare_partials("sigma", "A")

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["beam_force"] = 0
        residuals["beam_force"] += inputs["force0"]
        residuals["beam_force"] -= inputs["force1"]
        residuals['sigma'] = outputs['sigma'] - outputs['beam_force']/(1e6*inputs['A'])

    def solve_nonlinear(self, inputs, outputs):
        outputs['sigma'] =  outputs['beam_force']/(1e6*inputs['A'])

    def linearize(self, inputs, outputs, partials):
        partials["beam_force", "force0"] = 1
        partials["beam_force", "force1"] = -1
        partials["sigma", "beam_force"] = -1 / (1e6 * inputs["A"])
        partials["sigma", "sigma"] = 1
        partials["sigma", "A"] = outputs["beam_force"] / (1e6 * (inputs["A"]) ** 2)


class Node(ImplicitComponent):
 
    def initialize(self):
        self.options.declare("n_loads", default = 2, desc = "Number of loads on node")
        self.options.declare("n_external_forces", default = 0, desc = "Number of external forces on node")
        self.options.declare("n_reactions", default = 0, desc = "Number of reactions on node")

    def setup(self):

        for n in range(self.options["n_loads"]):

            n_load_out = f"load_out{n}"
            n_direction = f"direction{n}_load"
            n_load_in = f"load_in{n}"
            self.add_output(n_load_out, units = "N", desc = "Output load on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of load on node")
            self.add_input(n_load_in, units = "N", desc = "Input load on node")

        for n in range(self.options["n_reactions"]):

            n_reaction = f"reaction{n}"
            n_direction = f"direction{n}_reaction"
            self.add_output(n_reaction, units = "N", desc = "Output load on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of load on node")

        for m in range(self.options["n_external_forces"]):
            m_force = f"force{m}_ext"
            m_direction = f"direction{m}_ext"
            self.add_input(m_force, units = "N", desc = "External force on node")
            self.add_input(m_direction, units = "rad", desc = "Direction of external force on node")

        # for i in range(self.options["n_reactions"]):

        #     n_reaction = f"reaction{i}"
        #     self.declare_partials(n_load_out, "load_out*", method = "cs")
        #     self.declare_partials(n_load_out, "direction*", method = "cs")

        #     if (self.options["n_reactions"] > 0):

        #         self.declare_partials(n_load_out, "reaction*", method = "cs")
            
        #     if (self.options["n_external_forces"] > 0):

        #         self.declare_partials(n_load_out, "force*", method = "cs")

        # for n in range(2 - self.options["n_reactions"]):
        #     n_load_out = f"load_out{n}"
        #     self.declare_partials(n_load_out, "load_out*", method = "cs")
        #     self.declare_partials(n_load_out, "direction*", method = "cs")
            
        #     if (self.options["n_reactions"] > 0):

        #         self.declare_partials(n_load_out, "reaction*", method = "cs")
            
        #     if (self.options["n_external_forces"] > 0):

        #         self.declare_partials(n_load_out, "*ext", method = "cs")

        # for i in range((2 - self.options["n_reactions"]), self.options["n_loads"]):
        #     n_load_out = f"load_out{i}"
        #     n_load_in = f"load_in{i}"
        #     self.declare_partials(n_load_out, n_load_out, method = "cs")
        #     self.declare_partials(n_load_out, n_load_in, method = "cs")


        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):

        if (self.options["n_reactions"] > 0):
            
            if (self.options["n_reactions"] > 1):
                
                res0 = "reaction0"
                res1 = "reaction1"
            
            else:
                
                res0 = "reaction0"
                res1 = "load_out0"

        else:
            res0 = "load_out0"
            res1 = "load_out1"
            
        residuals[res0] = 0
        residuals[res1] = 0
        for n in range(self.options["n_loads"]):
            
            load = f"load_out{n}"
            direction = f"direction{n}_load"
            residuals[res0] += outputs[load] * np.cos(inputs[direction])
            residuals[res1] += outputs[load] * np.sin(inputs[direction])

        for j in range(self.options["n_reactions"]):

            reaction = f"reaction{j}"
            direction = f"direction{j}_reaction"
            residuals[res0] += outputs[reaction] * np.cos(inputs[direction])
            residuals[res1] += outputs[reaction] * np.sin(inputs[direction])
        
        for m in range(self.options["n_external_forces"]):
            force = f"force{m}_ext"
            direction = f"direction{m}_ext"
            residuals[res0] += inputs[force] * np.cos(inputs[direction])
            residuals[res1] += inputs[force] * np.sin(inputs[direction])

        
        for i in range((2 - self.options["n_reactions"]), self.options["n_loads"]):
            load_in = f"load_in{i}"
            load_out = f"load_out{i}"
            residuals[load_out] = outputs[load_out] - inputs[load_in]
            # print(self.pathname, load_out,  outputs[load_out] - inputs[load_in], residuals[load_out])

    # def linearize(self, inputs, outputs, partials):
        
    #     for n in range(self.options["n_loads"] + self.options["n_reactions"]):

    #         load = f"load{n}_out"
    #         direction = f"direction{n}"
    #         partials["load0_out", load] = np.cos(inputs[direction])
    #         partials["load0_out", direction] = -outputs[load] * np.sin(inputs[direction])
    #         partials["load1_out", load] = np.sin(inputs[direction])
    #         partials["load1_out", direction] = outputs[load] * np.cos(inputs[direction])

    #     for m in range(self.options["n_external_forces"]):
    #         force = f"force{m}_ext"
    #         direction = f"direction{m}_ext"
    #         partials["load0_out", force] = np.cos(inputs[direction])
    #         partials["load0_out", direction] = -inputs[force] * np.sin(inputs[direction])
    #         partials["load1_out", force] = np.sin(inputs[direction])
    #         partials["load1_out", direction] = inputs[force] * np.cos(inputs[direction])
        
    #     i = 2
    #     while (i < self.options["n_loads"]):
    #         load_out = f"load{i}_out"
    #         load_in = f"load{i}_in"
    #         direction = f"direction{i}"
    #         partials[load_out, load_out] = -inputs[load_in]
    #         partials[load_out, direction] = outputs[load_out]
    #         i += 1