import numpy as np
from openmdao.api import ExplicitComponent, ImplicitComponent, Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, NewtonSolver, DirectSolver, ScipyOptimizeDriver

class Beam(ImplicitComponent):

    def setup(self):
        # add inputs and outputs
        self.add_input("force0", val = 1., units = "N", desc = "Force on 0th end of beam")
        self.add_input("force1", val = 1., units = "N", desc = "Force on 1st end of beam")
        self.add_input("A", val = 1., units = "m**2", desc = "Cross sectional area of the beam")
        self.add_output("beam_force", val = 1., units = "N", desc = "Force in the beam")
        self.add_output('sigma', val=1, units='MPa')

        # declare necessary partials
        self.declare_partials("beam_force", "force*")
        self.declare_partials("sigma", "beam_force")
        self.declare_partials("sigma", "sigma")
        self.declare_partials("sigma", "A")

    def apply_nonlinear(self, inputs, outputs, residuals):
        # residuals balance forces on beam and compute stress
        residuals["beam_force"] = inputs["force0"] - inputs["force1"]
        residuals['sigma'] = outputs['sigma'] - outputs['beam_force']/(1e6*inputs['A'])

    def solve_nonlinear(self, inputs, outputs):
        # explicitly computes stress on beam
        outputs['sigma'] =  outputs['beam_force']/(1e6*inputs['A'])


    def linearize(self, inputs, outputs, partials):
        # analytic partial derivatives of beam_force residual
        partials["beam_force", "force0"] = 1
        partials["beam_force", "force1"] = -1
        # analytic partial derivatives of sigma residual
        partials["sigma", "beam_force"] = -1 / (1e6 * inputs["A"])
        partials["sigma", "sigma"] = 1
        partials["sigma", "A"] = outputs["beam_force"] / (1e6 * (inputs["A"]) ** 2)
    

class Node(ImplicitComponent):

    def initialize(self):
        # user specifies node configuration
        self.options.declare("n_loads", default = 2, desc = "Number of beams on node")
        self.options.declare("n_external_forces", default = 0, desc = "Number of external forces on node")
        self.options.declare("n_reactions", default = 0, desc = "Number of reactions on node")

    def setup(self):
        # add external forces and their directions as inputs
        for n in range(self.options["n_external_forces"]):
            n_force = f"force{n}_ext"
            n_direction = f"direction{n}_ext"
            self.add_input(n_force, units = "N", desc = "External force on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of external force on node")

        # add reaction forces as outputs, and their directions as inputs, and declare partials wrt external forces
        for m in range(self.options["n_reactions"]):
            n_reaction = f"reaction{m}"
            n_direction = f"direction{m}_reaction"
            self.add_output(n_reaction, units = "N", desc = "Output load on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of load on node")

            if (self.options["n_external_forces"] > 0):
                self.declare_partials(n_reaction, "force*")

        # add beams as load outputs and load inputs, add their directions as inputs
        for i in range(self.options["n_loads"]):
            n_load_out = f"load_out{i}"
            n_direction = f"direction{i}_load"
            n_load_in = f"load_in{i}"
            self.add_output(n_load_out, units = "N", desc = "Output load on node")
            self.add_input(n_direction, units = "rad", desc = "Direction of load on node")
            self.add_input(n_load_in, units = "N", desc = "Input load on node")

        # declare partials for reaction outputs
        if (self.options["n_reactions"] > 0):
            self.declare_partials("reaction*", "reaction*")
            self.declare_partials("reaction*", "load_out*")
            self.declare_partials("reaction*", "direction*")

        # declare partials for beams whose residuals have a force balance
        for j in range(2 - self.options["n_reactions"]):
            n_load_out = f"load_out{j}"
            self.declare_partials(n_load_out, "load_out*")
            self.declare_partials(n_load_out, "direction*")

            if (self.options["n_reactions"] > 0):
                self.declare_partials(n_load_out, "reaction*")

            if (self.options["n_external_forces"] > 0):
                self.declare_partials(n_load_out, "*ext")

        # declare partials for beams whose residuals do not have a force balance
        for k in range((2 - self.options["n_reactions"]), self.options["n_loads"]):
            n_load_out = f"load_out{k}"
            n_load_in = f"load_in{k}"
            self.declare_partials(n_load_out, n_load_out)
            self.declare_partials(n_load_out, n_load_in)


    def apply_nonlinear(self, inputs, outputs, residuals):
        # set the first two residuals to correspond with reactions if they exist, and beams if they don't
        res = ["string", "string"]
        for n in range(self.options["n_reactions"]):
            reaction = f"reaction{n}"
            res[n] = reaction   

        for m in range(0, 2 - self.options["n_reactions"]):
            load_out = f"load_out{m}"
            res[self.options["n_reactions"] - m] = load_out 

        # set initials value of first two residuals to zero
        residuals[res[0]] = 0
        residuals[res[1]] = 0

        # sum beam forces in x and y directions on node
        for i in range(self.options["n_loads"]):
            load = f"load_out{i}"
            direction = f"direction{i}_load"
            residuals[res[0]] += outputs[load] * np.cos(inputs[direction])
            residuals[res[1]] += outputs[load] * np.sin(inputs[direction])

        # sum reaction forces in x and y directions on node
        for j in range(self.options["n_reactions"]):
            reaction = f"reaction{j}"
            direction = f"direction{j}_reaction"
            residuals[res[0]] += outputs[reaction] * np.cos(inputs[direction])
            residuals[res[1]] += outputs[reaction] * np.sin(inputs[direction])

        # sum external forces in x and y directions on node
        for k in range(self.options["n_external_forces"]):
            force = f"force{k}_ext"
            direction = f"direction{k}_ext"
            residuals[res[0]] += inputs[force] * np.cos(inputs[direction])
            residuals[res[1]] += inputs[force] * np.sin(inputs[direction])

        # set output forces equal to inputs for remaining residuals
        for q in range((2 - self.options["n_reactions"]), self.options["n_loads"]):
            load_in = f"load_in{q}"
            load_out = f"load_out{q}"
            residuals[load_out] = outputs[load_out] - inputs[load_in]

    def linearize(self, inputs, outputs, partials):
        # identify first two residuals
        res = ["string", "string"]
        for n in range(self.options["n_reactions"]):
            reaction = f"reaction{n}"
            res[n] = reaction   

        for m in range(0, 2 - self.options["n_reactions"]):
            load_out = f"load_out{m}"
            res[self.options["n_reactions"] - m] = load_out 

        # compute partials for first two residuals wrt beams and their directions
        for i in range(self.options["n_loads"]):
            load = f"load_out{i}"
            direction = f"direction{i}_load"
            partials[res[0], load] = np.cos(inputs[direction])
            partials[res[0], direction] = -outputs[load] * np.sin(inputs[direction])
            partials[res[1], load] = np.sin(inputs[direction])
            partials[res[1], direction] = outputs[load] * np.cos(inputs[direction])

        # compute partials for first two residuals wrt reactions and their directions
        for j in range(self.options["n_reactions"]):
            reaction = f"reaction{j}"
            direction = f"direction{j}_reaction"
            partials[res[0], reaction] = np.cos(inputs[direction])
            partials[res[0], direction] = -outputs[reaction] * np.sin(inputs[direction])
            partials[res[1], reaction] = np.sin(inputs[direction])
            partials[res[1], direction] = outputs[reaction] * np.cos(inputs[direction])

        # compute partials for first two residuals wrt external forces and their directions
        for k in range(self.options["n_external_forces"]):
            force = f"force{k}_ext"
            direction = f"direction{k}_ext"
            partials[res[0], force] = np.cos(inputs[direction])
            partials[res[0], direction] = -inputs[force] * np.sin(inputs[direction])
            partials[res[1], force] = np.sin(inputs[direction])
            partials[res[1], direction] = inputs[force] * np.cos(inputs[direction])

        # compute partials for remaining residuals
        for q in range((2 - self.options["n_reactions"]), self.options["n_loads"]):
            load_out = f"load_out{q}"
            load_in = f"load_in{q}"
            partials[load_out, load_out] = 1
            partials[load_out, load_in] = -1