'''This script illustrates some problems in the definition of conditional distributions by the pgmpy package.
'''

from pgmpy.factors.discrete import TabularCPD

# Defining Conditional Probabilites
cpd_I = TabularCPD(variable='I', variable_card=2, values=[[1/4, 3/4]])
cpd_D = TabularCPD(variable='D', variable_card=2, values=[[1/4, 3/4]])
cpd_G = TabularCPD(variable='G', variable_card=2, values=[[0.5, 0.8, 0.75, 0.45],
                                                          [0.5, 0.2, 0.25, 0.55]],
                   evidence=['I', 'D'], evidence_card=[2, 2])

# printing the conditional distribution (P(G|I,D)
print(cpd_G)

# Illustrating that the marginalization in the conditional probability is wrong, without knowing the marginal.
# P(G|I) = integrate_over_D(P(G|I,D)*P(D|I)), but we don't know P(D|I)
wrong_marginal = cpd_G.marginalize(variables=['D'], inplace=False)
print(wrong_marginal)

# IF we knew that P(D|I) = P(D), then the marginalization is possible
correct_marginal = (cpd_G * cpd_D).marginalize(variables=['D'], inplace=False)
print(correct_marginal)

# correct_marginal != wrong marginal
print(wrong_marginal == correct_marginal)