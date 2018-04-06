import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Capture init_pose for use in reward function
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 0., 0., 0., 0.])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Capture provided reward function
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        """Reward design philosophy"""
        # The reward should increase as the copter gets to its target position (target_pos)
        # To to this, normalize the remaining distance by the total distance,
        # then subtract this value from 1, then multiply by a scaling factor
        # Punish rotations and x and y shifts from zero by applying a penalty
        """Calculating the reward"""
        # Calculate the distance between the current position(self.sim.pose[:3]) and target position
        distance_remaining = np.sqrt(np.sum([(x - y)**2 for x, y in zip(self.sim.pose[:3], self.target_pos)]))
        # Calculate the distance between the starting position and target position
        # This will be used to normalize the reward
        distance_total = np.sqrt(np.sum([(x - y)**2 for x, y in zip(self.init_pose[:3], self.target_pos)]))
        proximity = 1 - distance_remaining/distance_total
        proximity_reward = 10 * proximity if proximity > 0 else 0
        # Punish rotation
        rotation_punish = sum([1 if ii > 0.03 else 0 for ii in abs(self.sim.pose[3:])])
        # Punish shift
        shift_punish = sum([1 if ii > 1 else 0 for ii in abs(self.sim.pose[:2])])
        reward = proximity_reward - rotation_punish - shift_punish
        reward = reward/3 # rescale due to x3 action repeat
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state