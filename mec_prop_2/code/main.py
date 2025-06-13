"""
Main controller for Mechanical Properties 2 Lab Graphs
Runs all graphing modules and generates required PNGs.
"""
from bending_graphs import plot_bending_graphs
from annealing_impact_graphs import plot_annealing_impact_graphs
from dbtt_graphs import plot_dbtt_graph

def main():
    print("Generating graphs for Mechanical Properties 2 Lab...")
    plot_annealing_impact_graphs()
    plot_dbtt_graph()
    plot_bending_graphs()
    print("All graphs generated.")

if __name__ == "__main__":
    main()
