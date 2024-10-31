import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Spider:
    def __init__(self):
        # Body dimensions
        self.body_length = 10
        self.body_width = 6
        self.body_height = 4
        
        # Leg dimensions (same for all legs)
        self.L1 = 5  # Upper leg length
        self.L2 = 5  # Lower leg length
        
        # Body position and orientation
        self.body_pos = np.array([0, 0, 7])  # Starting height at 7cm
        self.body_orientation = 0  # Angle in radians
        
        # Calculate shoulder positions relative to body center
        self.shoulder_offsets = np.array([
            [self.body_length/2, self.body_width/2, 0],   # Front right
            [self.body_length/2, -self.body_width/2, 0],  # Front left
            [-self.body_length/2, self.body_width/2, 0],  # Back right
            [-self.body_length/2, -self.body_width/2, 0], # Back left
        ])
        
        # Initialize leg positions
        self.init_leg_positions()
    
    def init_leg_positions(self):
        """Initialize legs in default 90-degree pose"""
        self.leg_positions = []
        for shoulder_offset in self.shoulder_offsets:
            # Calculate world position of shoulder
            shoulder_pos = self.body_pos + shoulder_offset
            
            # Calculate default foot position (90-degree pose)
            foot_pos = shoulder_pos + np.array([
                self.L1 * (1 if shoulder_offset[0] > 0 else -1),  # Extend away from body
                self.L2 * (1 if shoulder_offset[1] > 0 else -1),
                -self.L1  # Down to ground
            ])
            self.leg_positions.append(foot_pos)
    
    def get_body_corners(self):
        """Get the corners of the body for visualization"""
        half_length = self.body_length / 2
        half_width = self.body_width / 2
        corners = np.array([
            [half_length, half_width, self.body_height/2],
            [half_length, -half_width, self.body_height/2],
            [-half_length, -half_width, self.body_height/2],
            [-half_length, half_width, self.body_height/2],
            [half_length, half_width, -self.body_height/2],
            [half_length, -half_width, -self.body_height/2],
            [-half_length, -half_width, -self.body_height/2],
            [-half_length, half_width, -self.body_height/2],
        ])
        # Transform corners to world coordinates
        return corners + self.body_pos

    def plot_spider(self, ax):
        """Plot the entire spider"""
        # Plot body
        corners = self.get_body_corners()
        
        # Plot body faces
        def plot_face(vertices):
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          color='blue', alpha=0.3)
        
        # Top face
        plot_face(corners[:4])
        # Bottom face
        plot_face(corners[4:])
        # Side faces
        for i in range(4):
            face = np.array([
                corners[i],
                corners[(i+1)%4],
                corners[(i+1)%4 + 4],
                corners[i + 4]
            ])
            plot_face(face)
        
        # Plot legs
        for i, (shoulder_offset, foot_pos) in enumerate(zip(self.shoulder_offsets, self.leg_positions)):
            shoulder_pos = self.body_pos + shoulder_offset
            
            # Calculate elbow position (90-degree angle)
            elbow_pos = shoulder_pos + np.array([
                self.L1 * (1 if shoulder_offset[0] > 0 else -1),
                self.L1 * (1 if shoulder_offset[1] > 0 else -1),
                0
            ])
            
            # Plot leg segments
            ax.plot([shoulder_pos[0], elbow_pos[0]], 
                   [shoulder_pos[1], elbow_pos[1]], 
                   [shoulder_pos[2], elbow_pos[2]], 'bo-')
            ax.plot([elbow_pos[0], foot_pos[0]], 
                   [elbow_pos[1], foot_pos[1]], 
                   [elbow_pos[2], foot_pos[2]], 'go-')

def animate(frame):
    ax.cla()
    spider.plot_spider(ax)
    
    # Set consistent view
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([0, 15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add ground plane
    xx, yy = np.meshgrid([-15, 15], [-15, 15])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

# Create figure and spider
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
spider = Spider()

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)
plt.show()
