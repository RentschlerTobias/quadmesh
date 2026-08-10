
from networkx import number_attracting_components
import torch
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, path=None):
        if path == None:
            path = "./saved_meshes/frame_field_time_measured/all_meshes_incl_time.pt"

        self.meshes = torch.load(path, weights_only=False)
        # self.grouped_data,self.bin_edges = self.group_meshes(self.meshes)
        # self.generate_box_plots(output_file="./figures/frame_field_time_numerical_v2.png")
        # self.generate_box_plots(eval_mode_gnn=True,output_file="./figures/frame_field_time_gnn_v2.png")
        self.plot_comparison()
#

    def group_meshes(self, list_of_meshes):

        num_intervals = 10

        data = []
        num_nodes = []

        # Init A list with num_intervals empy lists
        for i in range(num_intervals):
            new_list = []
            data.append(new_list)

        # Count number of nodes each mesh has
        for i in range(len(list_of_meshes)):
            num_nodes.append(list_of_meshes[i].x.size(0))

        # Generate n intervals in range from lowest to highest node count
        bin_edges = np.linspace(
            min(num_nodes)-1, max(num_nodes)+1, num_intervals+1)

        # Iterate through each mesh and add it to the list of the corresponding interval
        for i in range(len(list_of_meshes)):
            mesh = list_of_meshes[i]
            node_count = mesh.x.size(0)
            bin_index = np.digitize(node_count, bin_edges) - 1
            data[bin_index].append(mesh)

        return data, bin_edges

    def plot_comparison(self):
        import matplotlib.pyplot as plt

        num_nodes = []
        time_gnn = []
        time_num = []

        for mesh in self.meshes:
            if mesh.time_gnn < 0.2:
                num_nodes.append(mesh.x.size(0))
                time_gnn.append(mesh.time_gnn)
                time_num.append(mesh.time_frame_field_generator)

        plt.figure(figsize=(5, 6))
        plt.scatter(num_nodes, time_gnn, label="GNN", marker='o', color='b')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Computation Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./figures/time_comparison_gnn.png",
                    transparent=True, dpi=300)

        plt.figure(figsize=(5, 6))
        plt.scatter(num_nodes, time_num, label="Numeric",
                    marker='x', color='r')

        plt.xlabel('Number of Nodes')
        plt.ylabel('Computation Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("./figures/time_comparison_num.png",
                    transparent=True, dpi=300)

        plt.figure(figsize=(5, 6))
        plt.scatter(num_nodes, time_gnn, label="GNN", marker='o', color='b')
        plt.scatter(num_nodes, time_num, label="Numeric",
                    marker='x', color='r')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Computation Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./figures/time_comparison.png", transparent=True, dpi=300)

    def generate_box_plots(self, eval_mode_gnn=False, output_file="./figures/frame_field_time_bp.png"):
        # defaul eval mode is frame field generator else gnn can be used
        all_group_times = []
        group_labels = []

        for i in range(len(self.grouped_data)):
            current_data = self.grouped_data[i]
            if eval_mode_gnn == True:
                computation_times = [mesh.time_gnn for mesh in current_data]
            else:
                computation_times = [
                    mesh.time_frame_field_generator for mesh in current_data]
            all_group_times.append(computation_times)

            # Create label based on bin edges
            left = int(self.bin_edges[i])
            right = int(self.bin_edges[i + 1])
            group_labels.append(f"{left}-{right}")

        # Create the actual boxplot
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_group_times, labels=group_labels, showfliers=False)

        plt.xlabel("Node Count Interval")
        plt.ylabel("Computation Time (s)")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()

        plt.savefig(output_file, transparent=True, dpi=300)
