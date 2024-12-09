from sklearn import tree

modelo=tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

arbol = modelo.fit(X, y)

predicciones = arbol.predict(X)
accuracy_score(y, predicciones)

cross_val_score(arbol, X, y, cv=10).mean()
cross_val_score(arbol, X, y, cv=10, scoring='precision')
cross_val_score(arbol, X, y, cv=10, scoring='recall')

importanciasarbol=pd.DataFrame(arbol.feature_importances_)
importanciasarbol.index=(X.columns)
importanciasarbol



os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus

from sklearn import tree
tree.plot_tree(arbol, feature_names = X.columns, class_names=['0','1'], filled=True) 


dot_data = StringIO()

export_graphviz(arbol, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

import graphviz
gvz_graph = graphviz.Source(graph.to_string())
gvz_graph
gvz_graph.render("arbol", format="png", cleanup=True)