"""Simple illustration of variable elimination using the pgmpy package. The example that is used is the MontyHall.
"""
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

# Defining Conditional Probabilites
cpd_c = TabularCPD(variable='C', variable_card=3, values=[[1/3, 1/3, 1/3]])
cpd_p = TabularCPD(variable='P', variable_card=3, values=[[1/3, 1/3, 1/3]])
cpd_h = TabularCPD(variable='H', variable_card=3, values=[[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                                                          [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                                                          [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
                   evidence=['C', 'P'], evidence_card=[3, 3])


# Defining Joint through the product of the conditionals
# tmp is wrongly defined. It is a conditional, but its probabilities are of the joint.
# it is only used as an intermediate step to define the joint.
tmp = cpd_h * cpd_p * cpd_c
joint = JPD(variables=tmp.variables, cardinality=tmp.cardinality, values=tmp.values)

# check that computing the marginal from the joint is valid
assert(joint.marginal_distribution(variables=['C'], inplace=False) == cpd_c)
assert(joint.marginal_distribution(variables=['P'], inplace=False) == cpd_p)

# check that computing the conditional from the joint is valid
for i in range(2):
    for j in range(2):
        tmp1 = cpd_h.reduce([('C', i), ('P', j)], inplace=False)
        tmp2 = joint.conditional_distribution([('C', i), ('P', j)], inplace=False)
        assert(tmp1 == tmp2)



# we will infer the query P(P|C=1, H=0) with two methods
# (i) we define a Bayesian Model and perform variable elimination
# (ii) we will compute it straightly from the joint

# Define Bayesian Model (BM) and attach CPDs
BM = BayesianModel([['C', 'H'], ['P', 'H']])
BM.add_cpds(cpd_h, cpd_p, cpd_c)
assert BM.check_model()

# Infering the posterior probability on p with Variable Elimination
var_eliminator = VariableElimination(BM)
posterior_p_1 = var_eliminator.query(['P'], evidence={'C': 1, 'H': 0})
# type(posterior_p_1) -> factor

# Infering the posterior probability on p straightly from the Joint
posterior_p_2 = joint.conditional_distribution([('H', 0), ('C', 1)], inplace=False)
# type(posterior_p_2) -> JointProbabilityDistribution

assert("posterior_p_1 equals posterior_p_2 : %s" % (posterior_p_1 == posterior_p_2))


# perform 3 sampling methods
sampler = BayesianModelSampling(BM)
samples1 = sampler.forward_sample(size=10)
samples2 = sampler.likelihood_weighted_sample(size=10)
samples3 = sampler.rejection_sample(size=10)