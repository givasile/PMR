from pgmpy.readwrite import BIFReader
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
import networkx as nx
import matplotlib.pyplot as plt
import timeit

# Code Snippet for fetching the Model. Not needed if model is stored locally.
# import wget
# import gzip
# f = wget.download('http://www.bnlearn.com/bnrepository/asia/asia.bif.gz')
# with gzip.open('asia.bif.gz', mode='rb') as f:
#     file_content=f.read()
# with open('asia.bif', mode='wb') as f:
#     f.write(file_content)

# load model
reader = BIFReader('./saved_staff/asia.bif')
BM = reader.get_model()

# define joint as prod of conditionals
joint = BM.get_cpds()[0]
for cpd in BM.get_cpds()[1:]:
    joint = joint * cpd
joint = JPD(variables=joint.variables,
            cardinality=joint.cardinality,
            values=joint.values)

# variable elimination
a = timeit.default_timer()
var_eliminator = VariableElimination(BM)
q1 = var_eliminator.query(variables=['dysp'], evidence={'smoke': 'yes'})
print(q1)
print('Time for variable elimination: %.4f' % (timeit.default_timer() - a))

# belief propagation
a = timeit.default_timer()
belief_propagator = BeliefPropagation(BM)
q2 = belief_propagator.query(variables=['dysp'], evidence={'smoke': 'yes'})
q2.normalize()
print(q2)
print('Time for belief propagation  : %.4f' % (timeit.default_timer() - a))

# marginalization
a = timeit.default_timer()
cond = joint.conditional_distribution([('smoke', 0)], inplace=False)
joint = cond.marginalize(['asia', 'bronc', 'either', 'lung', 'tub', 'xray'], inplace=False)
print(joint)
print('Time for marginalization     : %.4f' % (timeit.default_timer() - a))
# joint.ma
# factor = joint.to_factor()

plt.figure()
nx.draw(BM, with_labels=True)
plt.show(block=False)
