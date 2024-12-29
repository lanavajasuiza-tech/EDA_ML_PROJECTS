from graphviz import Digraph

dot = Digraph(comment='Test Graph')
dot.node('A', 'Start')
dot.node('B', 'End')
dot.edge('A', 'B', 'Transition')

print(dot.source)
dot.render('Machine_Learning/graphviz-output/graph.png', format='png', cleanup=True)
