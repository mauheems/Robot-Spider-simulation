import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Spider:
    def __init__(self):
        # Body dimensions
        self.body_length = 6
        self.body_width = 6
        self.body_height = 2
        
        # Leg dimensions (same for all legs)
        self.L1 = 4 # Upper leg length
        self.L2 = 3  # Lower leg length
        
        # Body position and orientation
        self.body_pos = np.array([0, 0, 3])  # Starting height at 7cm
        self.body_orientation = 0  # Angle in radians
        
        # Calculate shoulder positions relative to body center
        self.shoulder_offsets = np.array([
            [self.body_length/2, self.body_width/2, 0],   # Front right
            [self.body_length/2, -self.body_width/2, 0],  # Front left
            [-self.body_length/2, self.body_width/2, 0],  # Back right
            [-self.body_length/2, -self.body_width/2, 0], # Back left
        ])
        
        # Gait parameters
        self.phase_offsets = [0, 0.5, 0.25, 0.75]  # Four-phase walking gait
        self.cycle_duration = 100  # Frames per cycle
        
        # Initialize leg positions and trajectories
        self.init_leg_positions()
        self.leg_trajectories = self.create_all_leg_trajectories()
    
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
    
    def create_leg_trajectory(self, shoulder_pos, forward_dir, side_dir):
        """Creates spline trajectory for a single leg including ground contact"""
        stride_length = 6
        stride_height = 3
        stride_width = 4
        
        # Define number of points for each phase
        num_swing_points = 20
        num_stance_points = 20
        total_points = num_swing_points + num_stance_points
        
        # Create time points
        t = np.linspace(0, 1, total_points)
        
        # Create points array
        points = np.zeros((total_points, 3))
        
        # Fill points array
        for i in range(total_points):
            normalized_t = t[i]
            
            if normalized_t < 0.4:  # Swing phase (40% of cycle)
                swing_t = normalized_t / 0.4
                
                # Parabolic height trajectory
                height = stride_height * 3 * swing_t * (1 - swing_t)
                
                # Side offset
                side_offset = stride_width * (1 + 0.7 * np.sin(np.pi * swing_t))
                
                # Forward movement (same direction for all legs)
                forward_pos = -stride_length/2 + stride_length * swing_t
                
            else:  # Stance phase (60% of cycle)
                stance_t = (normalized_t - 0.4) / 0.6
                height = 0
                side_offset = stride_width
                forward_pos = stride_length/2 - stride_length * stance_t
            
            # Calculate position
            pos = shoulder_pos.copy()
            pos += forward_dir * forward_pos  # Same forward direction for all legs
            pos += side_dir * side_offset
            pos[2] = height
            
            points[i] = pos
        
        # Create evaluation points
        t_new = np.linspace(0, 1, self.cycle_duration)
        
        # Create splines for each coordinate
        spline_x = CubicSpline(t, points[:, 0])
        spline_y = CubicSpline(t, points[:, 1])
        spline_z = CubicSpline(t, points[:, 2])
        
        x_traj = spline_x(t_new)
        y_traj = spline_y(t_new)
        z_traj = spline_z(t_new)
        
        return np.column_stack((x_traj, y_traj, z_traj))
    
    def create_all_leg_trajectories(self):
        """Creates trajectories for all legs"""
        trajectories = []
        
        for i, shoulder_offset in enumerate(self.shoulder_offsets):
            shoulder_pos = self.body_pos + shoulder_offset
            
            # Use same forward direction for all legs (positive X is forward)
            forward_dir = np.array([1, 0, 0])  # Same for all legs
            
            # Side direction depends on which side the leg is on
            side_dir = np.array([0, 1, 0]) if shoulder_offset[1] > 0 else np.array([0, -1, 0])
            
            trajectory = self.create_leg_trajectory(shoulder_pos, forward_dir, side_dir)
            trajectories.append(trajectory)
        
        return trajectories
    
    def update_leg_positions(self, frame):
        """Update leg positions based on current frame"""
        for i, trajectory in enumerate(self.leg_trajectories):
            # Apply phase offset
            phase = (frame + self.phase_offsets[i] * self.cycle_duration) % self.cycle_duration
            self.leg_positions[i] = trajectory[int(phase)]
    
    def calculate_leg_angles(self, shoulder_pos, target_pos):
        """Calculate angles for a leg with length constraints"""
        # Convert target to shoulder-relative coordinates
        relative_pos = target_pos - shoulder_pos
        x, y, z = relative_pos
        
        # Calculate distances
        R = np.sqrt(x**2 + y**2)  # Distance in xy plane
        D = np.sqrt(R**2 + z**2)  # Total distance
        
        # Check if target is reachable
        max_reach = self.L1 + self.L2
        if D > max_reach:
            # Scale the target position to maximum reach
            scale_factor = max_reach / D
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
            R = np.sqrt(x**2 + y**2)
            D = max_reach
        
        # Calculate angles
        theta_s = np.arctan2(y, x)  # Shoulder rotation
        theta_sp = 0  # Shoulder pitch (constrained to xy plane)
        
        # Calculate elbow angle
        cos_theta_e = (D**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta_e = np.clip(cos_theta_e, -1.0, 1.0)
        theta_e = np.arccos(cos_theta_e)
        
        return theta_s, theta_sp, theta_e

    def get_leg_positions(self, shoulder_pos, angles):
        """Get leg segment positions from angles"""
        theta_s, theta_sp, theta_e = angles
        
        # Calculate elbow position (constrained to xy plane)
        elbow_pos = shoulder_pos + np.array([
            self.L1 * np.cos(theta_s),
            self.L1 * np.sin(theta_s),
            0
        ])
        
        # Calculate foot position
        foot_pos = elbow_pos + np.array([
            self.L2 * np.cos(theta_s) * np.cos(theta_e),
            self.L2 * np.sin(theta_s) * np.cos(theta_e),
            -self.L2 * np.sin(theta_e)
        ])
        
        return elbow_pos, foot_pos

    def plot_spider(self, ax):
        """Plot the entire spider"""
        # Get body corners
        corners = self.get_body_corners()
        
        # Plot body edges (top)
        top_corners = corners[:4]
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([top_corners[i][0], top_corners[j][0]],
                    [top_corners[i][1], top_corners[j][1]],
                    [top_corners[i][2], top_corners[j][2]], 'b-')
        
        # Plot body edges (bottom)
        bottom_corners = corners[4:]
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([bottom_corners[i][0], bottom_corners[j][0]],
                    [bottom_corners[i][1], bottom_corners[j][1]],
                    [bottom_corners[i][2], bottom_corners[j][2]], 'b-')
        
        # Plot vertical edges
        for i in range(4):
            ax.plot([corners[i][0], corners[i+4][0]],
                    [corners[i][1], corners[i+4][1]],
                    [corners[i][2], corners[i+4][2]], 'b-')
        
        # Plot legs
        for i, (shoulder_offset, target_pos) in enumerate(zip(self.shoulder_offsets, self.leg_positions)):
            shoulder_pos = self.body_pos + shoulder_offset
            
            # Calculate angles
            angles = self.calculate_leg_angles(shoulder_pos, target_pos)
            
            # Get leg segment positions
            elbow_pos, foot_pos = self.get_leg_positions(shoulder_pos, angles)
            
            # Plot leg segments
            ax.plot([shoulder_pos[0], elbow_pos[0]], 
                   [shoulder_pos[1], elbow_pos[1]], 
                   [shoulder_pos[2], elbow_pos[2]], 'bo-', linewidth=2)
            ax.plot([elbow_pos[0], foot_pos[0]], 
                   [elbow_pos[1], foot_pos[1]], 
                   [elbow_pos[2], foot_pos[2]], 'go-', linewidth=2)
            
            # Plot target position if it's different from foot position
            if not np.allclose(target_pos, foot_pos):
                ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                          color='red', marker='x', s=100, alpha=0.5)
            
            # Plot trajectory for debugging
            trajectory = self.leg_trajectories[i]
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   'r--', alpha=0.2)

def animate(frame):
    ax.cla()
    
    # Update spider's leg positions
    spider.update_leg_positions(frame)
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
