import pybullet as p
import pybullet_data
import numpy as np
import time

class SpiderRobot:
    def __init__(self):
        # Initialize dimensions and parameters
        self.L1 = 0.04  # Upper leg length in meters
        self.L2 = 0.03  # Lower leg length in meters
        
        # Initialize simulation and load robot
        self.physicsClient = self.init_simulation()
        
        # Store joint IDs
        self.hip_joints = []
        self.knee_joints = []
        self.get_joint_ids()
        
        # Enable joint motor control
        self.enable_motors()

    def init_simulation(self):
        # Connect to PyBullet
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load the spider robot URDF
        self.robot_id = p.loadURDF(
            "spider_robot.urdf",
            basePosition=[0, 0, 0.05],
            useFixedBase=False
        )
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=0.3,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.05]
        )
        
        return physicsClient

    def get_joint_ids(self):
        # Get all joints
        for i in range(1, 5):  # For all 4 legs
            self.hip_joints.append(p.getJointInfo(self.robot_id, (i-1)*2)[0])    # Hip joints
            self.knee_joints.append(p.getJointInfo(self.robot_id, (i-1)*2+1)[0]) # Knee joints

    def enable_motors(self):
        # Enable motor control for all joints
        for hip, knee in zip(self.hip_joints, self.knee_joints):
            p.setJointMotorControl2(self.robot_id, hip, p.POSITION_CONTROL, 
                                  force=10, maxVelocity=3)
            p.setJointMotorControl2(self.robot_id, knee, p.POSITION_CONTROL, 
                                  force=10, maxVelocity=3)

    def set_leg_position(self, leg_index, hip_angle, knee_angle):
        """Set position for a specific leg"""
        # Invert knee angle for left legs (indices 1 and 3)
        if leg_index in [1, 3]:
            knee_angle = -knee_angle
            
        p.setJointMotorControl2(self.robot_id, self.hip_joints[leg_index], 
                              p.POSITION_CONTROL, hip_angle)
        p.setJointMotorControl2(self.robot_id, self.knee_joints[leg_index], 
                              p.POSITION_CONTROL, knee_angle)

    def get_leg_position(self, leg_index):
        """Get current position of a specific leg"""
        hip_state = p.getJointState(self.robot_id, self.hip_joints[leg_index])
        knee_state = p.getJointState(self.robot_id, self.knee_joints[leg_index])
        return hip_state[0], knee_state[0]  # Return current angles

def main():
    spider = SpiderRobot()
    
    try:
        while p.isConnected():
            # Move all legs to neutral position with outward stance
            for i in range(4):
                hip_angle = 0 if i in [1, 3] else -0  # Positive for left legs, negative for right legs
                knee_angle = 0.5  # This will be automatically inverted for left legs in set_leg_position
                spider.set_leg_position(i, hip_angle, knee_angle)
            
            p.stepSimulation()
            time.sleep(1./240.)  # 240 Hz
            
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()