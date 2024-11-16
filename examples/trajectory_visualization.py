"""
Semantic Trajectory Visualization Example

This script demonstrates the semantic momentum tracking system by visualizing:
1. Trajectory positions in reduced dimensional space
2. Confidence scores over time
3. Force field effects
4. Real-time text transcription
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import sounddevice as sd
from src.main import SemMomentSTT

class TrajectoryVisualizer:
    def __init__(
        self,
        semantic_dim=768,
        sample_rate=16000,
        history_length=50
    ):
        """
        Initialize the visualizer
        
        Args:
            semantic_dim: Dimensionality of semantic space
            sample_rate: Audio sample rate to use
            history_length: Number of past positions to display
        """
        self.stt = SemMomentSTT(semantic_dim=semantic_dim)
        self.pca = PCA(n_components=2)
        self.sample_rate = sample_rate
        self.history_length = history_length
        
        # Setup the plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 12))
        self.setup_subplots()
        
        # Initialize data storage
        self.positions = []
        self.confidences = []
        self.forces = []
        self.times = []
        self.texts = []
        self.current_time = 0
        
        # Plot objects
        self.trajectory_scatter = None
        self.force_quiver = None
        self.confidence_line = None
        
        plt.tight_layout()
    
    def setup_subplots(self):
        """Setup the visualization subplots"""
        # Trajectory plot
        self.ax_traj = self.fig.add_subplot(221)
        self.ax_traj.set_title('Semantic Space Trajectories (PCA)')
        self.ax_traj.set_xlabel('First Principal Component')
        self.ax_traj.set_ylabel('Second Principal Component')
        self.ax_traj.grid(True, alpha=0.3)
        
        # Force field plot
        self.ax_force = self.fig.add_subplot(222)
        self.ax_force.set_title('Semantic Force Field')
        self.ax_force.set_xlabel('First Principal Component')
        self.ax_force.set_ylabel('Second Principal Component')
        self.ax_force.grid(True, alpha=0.3)
        
        # Confidence plot
        self.ax_conf = self.fig.add_subplot(223)
        self.ax_conf.set_title('Trajectory Confidences')
        self.ax_conf.set_xlabel('Time (s)')
        self.ax_conf.set_ylabel('Confidence')
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.grid(True, alpha=0.3)
        
        # Text output plot
        self.ax_text = self.fig.add_subplot(224)
        self.ax_text.set_title('Transcribed Text')
        self.ax_text.axis('off')
    
    def _update_plot(self, frame):
        """Update the visualization"""
        # Generate audio frame
        duration = 0.1  # seconds
        samples = int(self.sample_rate * duration)
        audio_frame = np.random.randn(samples).astype(np.float32) * 0.1
        
        # Process frame
        result = self.stt.pipeline.process_frame(
            audio_frame,
            orig_sr=self.sample_rate
        )
        
        if result.trajectory is not None:
            # Update positions
            self.positions.append(result.trajectory.position)
            if len(self.positions) > self.history_length:
                self.positions.pop(0)
            
            # Update forces
            force = self.stt.pipeline.momentum_tracker.compute_force_field(
                result.trajectory.position
            )
            self.forces.append(force)
            if len(self.forces) > self.history_length:
                self.forces.pop(0)
            
            # Update confidences
            self.confidences.append(result.trajectory.confidence)
            self.times.append(self.current_time * duration)
            if len(self.confidences) > self.history_length:
                self.confidences.pop(0)
                self.times.pop(0)
            
            # Update text
            if result.text is not None:
                self.texts.append(f"[{self.current_time * duration:.1f}s] "
                                f"({result.confidence*100:.1f}%) {result.text}")
                if len(self.texts) > 10:  # Keep last 10 text segments
                    self.texts.pop(0)
            
            # Update trajectory plot
            positions_array = np.array(self.positions)
            if len(positions_array) > 1:
                # Project to 2D using PCA
                positions_2d = self.pca.fit_transform(positions_array)
                forces_2d = self.pca.transform(np.array(self.forces))
                
                # Update trajectory plot
                self.ax_traj.clear()
                self.ax_traj.set_title('Semantic Space Trajectories (PCA)')
                self.ax_traj.set_xlabel('First Principal Component')
                self.ax_traj.set_ylabel('Second Principal Component')
                self.ax_traj.grid(True, alpha=0.3)
                
                # Plot trajectory with confidence-based coloring
                scatter = self.ax_traj.scatter(
                    positions_2d[:, 0],
                    positions_2d[:, 1],
                    c=self.confidences,
                    cmap='viridis',
                    s=100,
                    alpha=0.6
                )
                
                # Plot connections between consecutive points
                for i in range(len(positions_2d) - 1):
                    self.ax_traj.plot(
                        positions_2d[i:i+2, 0],
                        positions_2d[i:i+2, 1],
                        'w-',
                        alpha=0.2
                    )
                
                # Update force field plot
                self.ax_force.clear()
                self.ax_force.set_title('Semantic Force Field')
                self.ax_force.set_xlabel('First Principal Component')
                self.ax_force.set_ylabel('Second Principal Component')
                self.ax_force.grid(True, alpha=0.3)
                
                # Plot force vectors
                self.ax_force.quiver(
                    positions_2d[:, 0],
                    positions_2d[:, 1],
                    forces_2d[:, 0],
                    forces_2d[:, 1],
                    color='r',
                    alpha=0.5
                )
                
                # Update confidence plot
                self.ax_conf.clear()
                self.ax_conf.set_title('Trajectory Confidences')
                self.ax_conf.set_xlabel('Time (s)')
                self.ax_conf.set_ylabel('Confidence')
                self.ax_conf.set_ylim(0, 1)
                self.ax_conf.grid(True, alpha=0.3)
                self.ax_conf.plot(self.times, self.confidences, 'g-', alpha=0.8)
                
                # Update text display
                self.ax_text.clear()
                self.ax_text.set_title('Transcribed Text')
                self.ax_text.axis('off')
                text_content = "\n".join(self.texts)
                self.ax_text.text(
                    0.05, 0.95, text_content,
                    fontsize=10, family='monospace',
                    verticalalignment='top',
                    transform=self.ax_text.transAxes
                )
            
            self.current_time += 1
    
    def run(self, duration=500):
        """Run the visualization"""
        print(f"Starting visualization... (Sample rate: {self.sample_rate}Hz)")
        print("Press Ctrl+C to stop.")
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=duration,
            interval=50,
            blit=False
        )
        
        plt.show()

def main():
    """Run the trajectory visualization"""
    print("Semantic Trajectory Visualization")
    print("================================")
    
    # Try different sample rates
    sample_rates = [16000, 44100]
    
    for rate in sample_rates:
        print(f"\nVisualizing with {rate}Hz audio...")
        visualizer = TrajectoryVisualizer(sample_rate=rate)
        try:
            visualizer.run()
        except KeyboardInterrupt:
            print("\nStopped visualization")
        plt.close('all')

if __name__ == "__main__":
    main()
