"""
Semantic Trajectory Visualization Example

This script demonstrates the semantic momentum tracking system by visualizing:
1. Trajectory positions in reduced dimensional space
2. Confidence scores over time
3. Force field effects
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from src.main import SemMomentSTT

class TrajectoryVisualizer:
    def __init__(self, semantic_dim=768):
        """Initialize the visualizer"""
        self.stt = SemMomentSTT(semantic_dim=semantic_dim)
        self.pca = PCA(n_components=2)
        
        # Setup the plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 12))
        self.fig.suptitle('Semantic Momentum Visualization', fontsize=16)
        
        # Trajectory plot
        self.ax1.set_title('Semantic Space Trajectories (PCA)')
        self.ax1.set_xlabel('First Principal Component')
        self.ax1.set_ylabel('Second Principal Component')
        self.ax1.grid(True, alpha=0.3)
        
        # Confidence plot
        self.ax2.set_title('Trajectory Confidences')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Confidence')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize data storage
        self.positions = []
        self.confidences = []
        self.times = []
        self.current_time = 0
        
        # Plot objects
        self.trajectory_scatter = None
        self.confidence_lines = []
        
        plt.tight_layout()
    
    def _update_plot(self, frame):
        """Update the visualization"""
        # Generate dummy audio frame (replace with real audio in practice)
        audio_frame = np.random.randn(16000).astype(np.float32)
        
        # Process frame
        trajectory = self.stt.pipeline.process_frame(audio_frame)
        
        if trajectory is not None:
            # Update positions
            self.positions.append(trajectory.position)
            if len(self.positions) > 50:  # Keep last 50 positions
                self.positions.pop(0)
            
            # Update confidences
            self.confidences.append(trajectory.confidence)
            self.times.append(self.current_time)
            if len(self.confidences) > 50:
                self.confidences.pop(0)
                self.times.pop(0)
            
            # Update trajectory plot
            positions_array = np.array(self.positions)
            if len(positions_array) > 1:
                # Project to 2D using PCA
                positions_2d = self.pca.fit_transform(positions_array)
                
                # Clear previous scatter
                if self.trajectory_scatter:
                    self.trajectory_scatter.remove()
                
                # Plot new positions with confidence-based coloring
                self.trajectory_scatter = self.ax1.scatter(
                    positions_2d[:, 0],
                    positions_2d[:, 1],
                    c=self.confidences,
                    cmap='viridis',
                    s=100,
                    alpha=0.6
                )
                
                # Plot connections between consecutive points
                for i in range(len(positions_2d) - 1):
                    self.ax1.plot(
                        positions_2d[i:i+2, 0],
                        positions_2d[i:i+2, 1],
                        'w-',
                        alpha=0.2
                    )
            
            # Update confidence plot
            self.ax2.clear()
            self.ax2.set_title('Trajectory Confidences')
            self.ax2.set_xlabel('Time')
            self.ax2.set_ylabel('Confidence')
            self.ax2.set_ylim(0, 1)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.plot(self.times, self.confidences, 'g-', alpha=0.8)
            
            self.current_time += 1
        
        return self.trajectory_scatter,
    
    def run(self, duration=500):
        """Run the visualization"""
        print("Starting visualization... Press Ctrl+C to stop.")
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=duration,
            interval=50,
            blit=True
        )
        
        plt.show()

def main():
    """Run the trajectory visualization"""
    print("Semantic Trajectory Visualization")
    print("================================")
    
    visualizer = TrajectoryVisualizer()
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nStopped visualization")

if __name__ == "__main__":
    main()
