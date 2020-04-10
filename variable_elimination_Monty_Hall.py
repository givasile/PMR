"""Monty Hall example, using Variable Elimination.
"""
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Defining Joint as product of CPDs
cpd_c = TabularCPD('C', 3, [[1/3, 1/3, 1/3]])
cpd_p = TabularCPD('P', 3, [[1/3, 1/3, 1/3]])
cpd_h = TabularCPD('H', 3, [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                            [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                            [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
                   evidence=['C', 'P'], evidence_card=[3, 3])
joint_as_CPD = cpd_h * cpd_p * cpd_c

# Defining Joint as factor
joint_as_factor = joint_as_CPD.to_factor()

# Defining Joint with the default method
joint1 = JPD(variables=joint_as_CPD.variables,
             cardinality=joint_as_CPD.cardinality,
             values=joint_as_CPD.values)

# Define Bayesian Model (BM) and attach CPDs
BM = BayesianModel([['C', 'H'], ['P', 'H']])
BM.add_cpds(cpd_h, cpd_p, cpd_c)
print(BM.check_model())

# Infering the posterior probability on p with Variable Elimination
var_eliminator = VariableElimination(BM)
posterior_p_1 = var_eliminator.query(['P'], evidence={'C': 1, 'H': 0})
print(posterior_p_1)

# Infering the posterior probability on p straightly from the Joint
posterior_p_2 = joint1.conditional_distribution([('H', 0), ('C', 1)], inplace=False)
print(posterior_p_2)


print("posterior_p_1 equals posterior_p_2 : %s" % (posterior_p_1 == posterior_p_2))


plt.figure()
nx.draw(BM, with_labels=True)
plt.show()
