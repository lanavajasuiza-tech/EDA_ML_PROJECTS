import graphviz

# Crear un objeto Source
dot = """
    digraph G {
        A -> B;
        B -> C;
        A -> C;
    }
"""

gvz_graph = graphviz.Source(dot)
gvz_graph.view()  # Esto debería abrir el gráfico en un visor de PDF o en un navegador, según la configuración.
