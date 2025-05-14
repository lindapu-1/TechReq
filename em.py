import json
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

class MetricsVisualizer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='mps'):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        
    # def load_metrics(self, json_file):
    #     """Load metrics and units"""
    #     with open(json_file, 'r') as f:
    #         data = json.load(f)
            
    #     self.all_metrics = []
    #     self.units = []
            # Use set to remove duplicates
        # metrics_set = set()
        # for metrics_list in data.values():
        #     for metric, unit in metrics_list:
        #         if metric not in metrics_set:
        #             metrics_set.add(metric)
        #             self.all_metrics.append(metric)
        #             self.units.append(unit)
                    
        # return self.all_metrics, self.units

    def load_metrics(self, json_file):
        """Load metrics and units, while saving the corresponding needs information"""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        self.all_metrics = []
        self.units = []
        self.metric_to_needs = {}  # New: Store needs corresponding to metrics
        
        metrics_set = set()
        for need, metrics_list in data.items():
            for metric, unit in metrics_list:
                if metric not in metrics_set:
                    metrics_set.add(metric)
                    self.all_metrics.append(metric)
                    self.units.append(unit)
                    self.metric_to_needs[metric] = []
                # Add need to the corresponding metric's needs list
                self.metric_to_needs[metric].append(need)
                    
        return self.all_metrics, self.units

    def compute_embeddings(self):
        """Compute embeddings for metrics"""
        self.embeddings = self.model.encode(self.all_metrics, device=self.device)
        return self.embeddings
    
    def find_similar_pairs(self, similarity_threshold=0.75):
        """Find similar pairs of metrics, including corresponding needs information"""
        similar_pairs = []
        for i in range(len(self.all_metrics)):
            for j in range(i + 1, len(self.all_metrics)):
                similarity = 1 - cosine(self.embeddings[i], self.embeddings[j])
                if similarity > similarity_threshold:
                    similar_pairs.append({
                        'metric1': self.all_metrics[i],
                        'metric2': self.all_metrics[j],
                        'similarity': similarity,
                        'unit1': self.units[i],
                        'unit2': self.units[j],
                        'needs1': self.metric_to_needs[self.all_metrics[i]],
                        'needs2': self.metric_to_needs[self.all_metrics[j]]
                    })
    
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_pairs

    def reduce_dimensions(self):
        """Use t-SNE for dimensionality reduction"""
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42)
        self.embeddings_2d = tsne.fit_transform(self.embeddings)
        return self.embeddings_2d
    
    def visualize(self):
        """Create an interactive visualization with labels"""
        embeddings_2d = self.reduce_dimensions()
        
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'metric': self.all_metrics,
            'unit': self.units
        })
        
        # Create a basic scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers+text',  # Show points and text
            text=df['metric'],    # Show metric names
            textposition="top center",  # Text position
            hovertemplate="<br>".join([
                "Metric: %{text}",
                "Unit: %{customdata}",
                "<extra></extra>"
            ]),
            customdata=df['unit'],
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title="Metrics Semantic Similarity Visualization",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            showlegend=False,
            # Adjust the size and margins of the plot to fit labels
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Optimize text overlap
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        
        return fig
    
    def print_similar_pairs(self, similar_pairs, similarity_threshold=0.75):
        print(f"\nSimilar Metrics Pairs (similarity > {similarity_threshold}):")
        for i, pair in enumerate(similar_pairs, 1):
            print(f"\nPair {i}:")
            print(f"Metric 1: {pair['metric1']} ({pair['unit1']})")
            print(f"Metric 2: {pair['metric2']} ({pair['unit2']})")
            print(f"Similarity: {pair['similarity']:.3f}")
            
            # Convert needs to a set for comparison
            needs1_set = set(pair['needs1'])
            needs2_set = set(pair['needs2'])
            
            if needs1_set == needs2_set:
                # If needs are exactly the same
                print(f"Same need: {', '.join(needs1_set)}")
            else:
                # If needs are different
                print(f"Need 1: {', '.join(pair['needs1'])}")
                print(f"Need 2: {', '.join(pair['needs2'])}")
            
            print("-" * 80)  # Separator line


def main():
    visualizer = MetricsVisualizer()
    
    # Load data
    print("Loading metrics data...")
    metrics, units = visualizer.load_metrics('results/20250410_121451/final_metrics.json')
    print(f"Loaded {len(metrics)} unique metrics")
    
    # Compute embeddings
    print("Computing embeddings...")
    visualizer.compute_embeddings()
    
    # Find similar pairs
    print("\nFinding similar metrics...")
    similarity_threshold = 0.6
    similar_pairs = visualizer.find_similar_pairs(similarity_threshold=similarity_threshold)
    
    # Print similar pairs
    visualizer.print_similar_pairs(similar_pairs, similarity_threshold=similarity_threshold)

    # Visualization
    print("\nGenerating visualization...")
    fig = visualizer.visualize()
    fig.write_html('results/20250410_121451/metrics_similarity_visualization.html')
    fig.show()


if __name__ == "__main__":
    main()
