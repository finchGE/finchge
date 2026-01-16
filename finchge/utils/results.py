
import csv
import statistics
import logging
from finchge.utils.loghelper import get_log_dir, get_logger
import matplotlib
matplotlib.use('Agg')  # Switch to non-GUI backend
import matplotlib.pyplot as plt
from finchge.grammar.tree_node import TreeNode
import plotly.graph_objects as go
import plotly.colors as colors
from tabulate import tabulate
import pandas as pd
import numpy as np
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class ResultHelper:
    def __init__(self, project_id="finch"):
        self.project_id = project_id
        self.logger = get_logger(project_id)

    def generate_summary(self, objective_names):
        """
        Generate and save summary statistics for multi-objective genetic evolution.

        Args:
            objective_names (list): Names of the objective functions
        """
        # read from generations.csv
        csv_file_path = f"{get_log_dir(self.project_id)}/generations.csv"
        fitness_data = {name: [] for name in objective_names}
        used_codons_data = []
        tree_depths = []
        generations = []

        # read csv
        try:
            with open(csv_file_path, mode='r') as file:
                reader = csv.DictReader(file)  # Standard CSV format
                for row in reader:
                    generations.append(int(row['generation']))

                    # all  objectives
                    for objective_name in objective_names:
                        try:
                            fitness_data[objective_name].append(float(row[objective_name]))
                        except (ValueError, KeyError):
                            self.logger.warning(f"Parsing failed for {objective_name}, set to 0")
                            fitness_data[objective_name].append(0)

                    #  used codons
                    try:
                        used_codons_data.append(int(row['used_codons']))
                    except (ValueError, KeyError):
                        self.logger.warning(f"Parsing failed for used_codons in generation {row.get('generation', 'unknown')}, set to -1")
                        used_codons_data.append(-1)

                    #  tree depth
                    try:
                        tree_depths.append(int(row['tree_depth']))
                    except (ValueError, KeyError):
                        self.logger.warning(
                            f"Parsing failed for tree_depth in generation {row.get('generation', 'unknown')}, set to 0")
                        tree_depths.append(0)
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {csv_file_path}")
            return

        # Ensure we have data to process
        if not all(fitness_data.values()) or not used_codons_data or not tree_depths:
            self.logger.error("Not enough data to generate summary")
            return

        # Calculate statistics
        max_fitness = {name: round(max(values), 4) if values else 0 for name, values in fitness_data.items()}
        min_fitness = {name: round(min(values), 4) if values else 0 for name, values in fitness_data.items()}
        avg_fitness = {name: round(statistics.mean(values), 4) if values else 0 for name, values in
                       fitness_data.items()}
        std_fitness = {name: round(statistics.stdev(values), 4) if len(values) > 1 else 0 for name, values in
                       fitness_data.items()}

        # Prepare tables
        headers = ["Metric", *objective_names]
        fitness_stats = [
            ["Max", *[max_fitness[name] for name in objective_names]],
            ["Min", *[min_fitness[name] for name in objective_names]],
            ["Average", *[avg_fitness[name] for name in objective_names]],
            ["Std Dev", *[std_fitness[name] for name in objective_names]]
        ]

        other_stats = [
            ["Used Codons", min(used_codons_data) if used_codons_data else 0,
             max(used_codons_data) if used_codons_data else 0],
            ["Tree Depth", min(tree_depths) if tree_depths else 0, max(tree_depths) if tree_depths else 0]
        ]
        other_headers = ["Metric", "Min", "Max"]

        # tabulate summary text
        summary = "\nFitness Statistics:\n"
        summary += tabulate(fitness_stats, headers=headers, tablefmt="pretty")
        summary += "\n\nOther Statistics:\n"
        summary += tabulate(other_stats, headers=other_headers, tablefmt="pretty")

        # print and save summary
        print(summary)
        self.logger.info(summary)
        summary_path = f"{get_log_dir(self.project_id)}/summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        self.logger.info(f"Summary saved to {summary_path}")

        # fitness chart for single objective optimization only.
        if len(objective_names) == 1:
            fitness_data = fitness_data[objective_names[0]]  # one fitness in the list
            if fitness_data:
                self.plot_fitness_chart(fitness_data)
                best_fitness_idx = fitness_data.index(max(fitness_data))
                best_generation = generations[best_fitness_idx]

                # tree for best generation

                best_tree_path = f"{get_log_dir(self.project_id)}/trees/{best_generation}.json"
                try:
                    with open(best_tree_path, "r") as file:
                        best_tree_json = file.read()
                    self.plot_tree(best_tree_json)
                except FileNotFoundError:
                    self.logger.info(f"Skipping plotting phenotype mapping tree: Tree file for generation {best_generation} not found. Please ensure logging is not excluded for tree files in config.")


    def visualize_pareto_front(self, objective_names):
        pareto_csv_path = f"{get_log_dir(self.project_id)}/generations.csv"
        df = pd.read_csv(pareto_csv_path)
        # Create base directory for plots
        os.makedirs(f"{get_log_dir(self.project_id)}", exist_ok=True)

        # Get sorted generations
        unique_gens = sorted(df['generation'].unique())
        fig, ax1 = plt.subplots(figsize=(10, 8))
        # define colors
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_gens)))

        # TODO check this later
        step_dir = 'pre'  # Change to 'post' if minimizing

        # all solutions with transparency
        all_data = df[objective_names + ['generation']]
        plt.scatter(
            all_data[objective_names[0]],
            all_data[objective_names[1]],
            c=all_data['generation'],
            cmap='viridis',
            alpha=0.2,
            s=15,
            marker='.'
        )

        # Plot each generation
        for i, gen in enumerate(unique_gens):
            gen_data = df[df['generation'] == gen]
            gen_data_sorted = gen_data.sort_values(by=objective_names[0])

            ax1.step(gen_data_sorted[objective_names[0]],
                     gen_data_sorted[objective_names[1]],
                     linestyle='--',
                     where=step_dir,
                     color=colors[i],
                     lw=0.35,
                     alpha=0.25)
            ax1.plot(gen_data_sorted[objective_names[0]],
                     gen_data_sorted[objective_names[1]],
                     'o',
                     color=colors[i],
                     ms=1)
        ax1.set_xlabel(objective_names[0], fontsize=14)
        ax1.set_ylabel(objective_names[1], fontsize=14)
        plt.title("Pareto Fronts by Generation")

        # color map
        norm = Normalize(vmin=0, vmax=len(unique_gens) - 1)
        sm = ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])

        # colorbar
        cbar = plt.colorbar(sm, ax=ax1, ticks=[0, len(unique_gens) - 1])
        cbar.ax.set_ylabel('Generation', rotation=90)

        # Save plot as pdf and image
        plt.tight_layout()
        plt.savefig(f"{get_log_dir(self.project_id)}/pareto_front_evolution.pdf")
        plt.savefig(f"{get_log_dir(self.project_id)}/pareto_front_evolution.png", dpi=300)
        plt.close()
        self.logger.info(f"Pareto front evolution plot saved to: {get_log_dir(self.project_id)}/pareto_front_evolution.pdf")

    def plot_fitness_chart(self, data):
        plt.plot(data)
        plt.title('Fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness value')
        plt.savefig(f"{get_log_dir(self.project_id)}/fitness_chart.png")
        plt.close()
        self.logger.info(f"Fitness chart saved to {get_log_dir(self.project_id)}/fitness_chart.png")



    def plot_tree(self, tree_json):
        tree = TreeNode.from_json(tree_json)

        # Lists to store node info
        nodes_x = []  # x coordinates
        nodes_y = []  # y coordinates
        nodes_text = []  # node labels
        nodes_color = [] # node color
        edge_x = []  # x coordinates for edges
        edge_y = []  # y coordinates for edges

        colorscale = colors.sequential.Viridis

        def process_node(node, x, y, level=0):
            nodes_x.append(x)
            nodes_y.append(y)
            nodes_text.append(str(node.symbol))

            color_idx = level % len(colorscale)
            nodes_color.append(colorscale[color_idx])

            # Process children
            n_children = len(node.children)
            if n_children > 0:
                spacing = 2.0 / (n_children + 1)
                for i, child in enumerate(node.children):
                    child_x = x + 1
                    child_y = y - 1 + (i + 1) * spacing

                    # Add edge coordinates
                    edge_x.extend([x, child_x, None])
                    edge_y.extend([y, child_y, None])

                    process_node(child, child_x, child_y, level + 1)

        # Start from root
        process_node(tree, 0, 0)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers+text',
            text=nodes_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=20,
                color=nodes_color,  # Use our color list
                line_width=2,
                line=dict(color='black')  # Add black borders to nodes
            ))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        # Save tree as html
        fig.write_html(f"{get_log_dir(self.project_id)}/fitness_tree.html")
        #fig.write_image(f"{get_log_dir(self.project_id)}/fitness_tree.jpeg") # Not working on windows

    def save_pareto_front(self, pareto_front, objective_names):
        pareto_csv_path = f"{get_log_dir()}/pareto_front.csv"

        # fitness objectives + phenotype
        headers = objective_names + ['phenotype']

        with open(pareto_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for ind in pareto_front:
                row = list(ind.fitness) + [ind.phenotype]
                writer.writerow(row)

        self.visualize_pareto_front(objective_names)

