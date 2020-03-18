import numpy as np
from collections import deque

class Environment():
    """ Environment for walking task
        Defines and builds states, applies actions, and provides rewards to
        robot. 
        Use step() function to execute a timestep in the environment
        Use reset() function to initialize and reset state
    """

    def __init__(self, robot):

        print("Starting Environment...")
        self.state = None
        self.robot = robot

        # Webots custom data field is used to communicate with supervisor node
        self.robot.setCustomData('')

        # Get transform nodes by name
        self.left_foot = robot.getFromDef('LeftF')
        self.right_foot = robot.getFromDef('RightF')
        self.robot_center = robot.getFromDef('CoG')

        # tried using multiple states, ignored by using value 1
        self.history_size = 1
        self.state_buffer = deque(maxlen=self.history_size)

        self.timestep = int(robot.getBasicTimeStep())

        self.gps = robot.getGPS('gps')
        self.gps.enable(self.timestep)

        # intialize position, distance, and timestep counter
        self.last_position = 0
        self.total_distance = 0
        self.old_distance_reward = 0
        self.iternation = 0

        # storage for reward sub-objectives, values are send to tensorboard
        self.total_reward_contents = np.zeros((9,), dtype=np.float32)

                
        # values from https://www.cyberbotics.com/doc/guide/nao
        # NAO Left Foot Motors
        self.l_ankle_pitch = robot.getMotor('LAnklePitch') # (-1.18 to 0.92)
        self.l_ankle_roll = robot.getMotor('LAnkleRoll')  # (-0.76 to 0.39) ??
        self.l_hip_pitch = robot.getMotor('LHipPitch') # (-1.77 to 0.48)
        self.l_hip_roll = robot.getMotor('LHipRoll') # (-0.37 to 0.79)
        self.l_hip_yaw_pitch = robot.getMotor('LHipYawPitch') # (-1.14 to 0.74)
        self.l_knee_pitch = robot.getMotor('LKneePitch') # (-0.09 to 2.11)

        # NAO Right Foot Motors
        self.r_ankle_pitch = robot.getMotor('RAnklePitch') # (-1.18 to 0.92)
        self.r_ankle_roll = robot.getMotor('RAnkleRoll') # (-0.38 to 0.39) ??
        self.r_hip_pitch = robot.getMotor('RHipPitch') # (-1.77 to 0.48)
        self.r_hip_roll = robot.getMotor('RHipRoll') # (-0.37 to 0.44)
        self.r_hip_yaw_pitch = robot.getMotor('RHipYawPitch') # (-1.14 to 0.74)
        self.r_knee_pitch = robot.getMotor('RKneePitch') # (-0.09 to 2.11)       

        # Used motors
        motor_names = ['LAnklePitch', 'LHipPitch', 'LKneePitch', 'RAnklePitch',
            'RHipPitch',  'RKneePitch', 'LShoulderPitch', 'RShoulderPitch']

        self.motors = []
        self.action_bounds = []
        self.initial_position = []
        self.old_positions = []
        self.upper_bounds = []

        # get motors and bounds of actions
        for name in motor_names:
            # load motor
            motor = robot.getMotor(name)
            self.motors.append(motor)

            # set new max velocity to lower value            
            # max_velocitiy = motor.getMaxVelocity()
            # motor.setVelocity(max_velocitiy * 0.75)

            # get bounds of allowed actions
            min_action = motor.getMinPosition()
            max_action = motor.getMaxPosition()
            self.upper_bounds.append(np.maximum(np.abs(min_action), np.abs(max_action)))
            self.action_bounds.append([min_action, max_action])

            # get initial position info
            init_position = motor.getTargetPosition()
            self.initial_position.append(init_position)
            self.old_positions.append(0.0)
            

        # Enable NAO motors ans sensors
        # NAO Hands
        self.l_shoulder_pitch = robot.getMotor('LShoulderPitch')
        self.r_shoulder_pitch = robot.getMotor('RShoulderPitch')

        # NAO accelerometer
        self.accelerometer = robot.getAccelerometer('accelerometer')
        self.accelerometer.enable(self.timestep)

        # NAO gyroscope
        self.gyro = robot.getGyro('gyro')
        self.gyro.enable(self.timestep)

        # NAO inertial unit
        self.inertial_unit = robot.getInertialUnit('inertial unit')
        self.inertial_unit.enable(self.timestep)

        # NAO position sensors
        self.l_ankle_pitch_pos = robot.getPositionSensor('LAnklePitchS')
        self.l_ankle_pitch_pos.enable(self.timestep)
        # self.l_ankle_roll_pos = robot.getPositionSensor('LAnkleRollS')
        # self.l_ankle_roll_pos.enable(self.timestep)
        self.l_hip_pitch_pos = robot.getPositionSensor('LHipPitchS')
        self.l_hip_pitch_pos.enable(self.timestep)
        # self.l_hip_roll_pos = robot.getPositionSensor('LHipRollS')
        # self.l_hip_roll_pos.enable(self.timestep)
        # self.l_hip_yaw_pitch_pos = robot.getPositionSensor('LHipYawPitchS')
        # self.l_hip_yaw_pitch_pos.enable(self.timestep)
        self.l_knee_pitch_pos = robot.getPositionSensor('LKneePitchS')
        self.l_knee_pitch_pos.enable(self.timestep)

        self.r_ankle_pitch_pos = robot.getPositionSensor('RAnklePitchS')
        self.r_ankle_pitch_pos.enable(self.timestep)
        # self.r_ankle_roll_pos = robot.getPositionSensor('RAnkleRollS')
        # self.r_ankle_roll_pos.enable(self.timestep)
        self.r_hip_pitch_pos = robot.getPositionSensor('RHipPitchS')
        self.r_hip_pitch_pos.enable(self.timestep)
        # self.r_hip_roll_pos = robot.getPositionSensor('RHipRollS')
        # self.r_hip_roll_pos.enable(self.timestep)
        # self.r_hip_yaw_pitch_pos = robot.getPositionSensor('RHipYawPitchS')
        # self.r_hip_yaw_pitch_pos.enable(self.timestep)
        self.r_knee_pitch_pos = robot.getPositionSensor('RKneePitchS')
        self.r_knee_pitch_pos.enable(self.timestep)

        # Shoulder position
        self.r_shoulder_pitch_pos = robot.getPositionSensor('RShoulderPitchS')
        self.r_shoulder_pitch_pos.enable(self.timestep)
        self.l_shoulder_pitch_pos = robot.getPositionSensor('LShoulderPitchS')
        self.l_shoulder_pitch_pos.enable(self.timestep)

    # change motor postion depending on given action vector
    def _act(self, action_vector):
        # convert to double, setPostion does not like floats
        action_vector = np.array(action_vector, dtype=np.float64)
        for i in range(len(action_vector)):
            # set new motor position, overrides previous value
            self.motors[i].setPosition(action_vector[i])
            
    # build state vector from values provided by position sensors
    def _build_state(self):
        state = []

        # timestep
        time_step = self.iternation * 0.0032 * 4    # {n} * frequency * action repeat
        state.append(time_step)

        # acceleration
        accel_values = self.accelerometer.getValues()
        for i in accel_values:
            state.append(i)

        # orientation, ignore 3rd value: NaN
        gyro_values = self.gyro.getValues()
        state.append(gyro_values[0])
        state.append(gyro_values[1])
        # print(gyro_values)

        # angular velocities, ignore 3rd value: NaN 
        inertial_unit_values = self.inertial_unit.getRollPitchYaw() 
        state.append(inertial_unit_values[0])                               
        state.append(inertial_unit_values[1])

        # torso position in world space, relatve to start [0,0,0]
        #gps_values = self.gps.getValues() 
        #state.append(gps_values[0])          
        #state.append(gps_values[1])
        #state.append(gps_values[2])

        # left foot position in world space
        #left_foot_position = np.array(self.left_foot.getPosition())
        #state.append(left_foot_position[0])
        #state.append(left_foot_position[1])
        #state.append(left_foot_position[2])

        # right foot position in world space 
        #right_foot_position = np.array(self.right_foot.getPosition())
        #state.append(right_foot_position[0])
        #state.append(right_foot_position[1])
        #state.append(right_foot_position[2])

        # left leg position sensors
        pos_l_ankle_pitch = self.l_ankle_pitch_pos.getValue()
        state.append(pos_l_ankle_pitch)
        pos_l_hip_pitch = self.l_hip_pitch_pos.getValue()
        state.append(pos_l_hip_pitch) 
        pos_l_knee_pitch = self.l_knee_pitch_pos.getValue()
        state.append(pos_l_knee_pitch) 

        # right leg position sensors
        pos_r_ankle_pitch = self.r_ankle_pitch_pos.getValue()
        state.append(pos_r_ankle_pitch)
        pos_r_hip_pitch = self.r_hip_pitch_pos.getValue()
        state.append(pos_r_hip_pitch) 
        pos_r_knee_pitch = self.r_knee_pitch_pos.getValue()
        state.append(pos_r_knee_pitch) 

        # sholder positions
        pos_r_shoulder_pitch = self.r_shoulder_pitch_pos.getValue()
        state.append(pos_r_shoulder_pitch)
        pos_l_shoulder_pitch = self.l_shoulder_pitch_pos.getValue()
        state.append(pos_l_shoulder_pitch)

        return state

    # clip predicted action in allowed action bounds
    def _transform_action(self, action_vector):

        # action_vector = np.clip(action_vector, -1, 1)
        original_actions = np.copy(action_vector)
        rescaled_actions = np.copy(action_vector)

        for i in range(len(action_vector)):
            action = action_vector[i]
            min_bound = self.action_bounds[i][0] + 1e-4
            max_bound = self.action_bounds[i][1] - 1e-4
            # upper_bound = self.upper_bounds[i]
            # action = rescale_in_range(action, a=min_bound, b=max_bound)
            action = np.clip(action, min_bound, max_bound)
            rescaled_actions[i] = action

        # print(rescaled_actions)
        return original_actions, rescaled_actions

    # build initial state s_0
    def reset(self):
        self.state = self._build_state()

        # ignore, not longer relevant
        for _ in range(self.history_size):
            self.state_buffer.append(self.state)

        state = np.array(self.state_buffer, dtype=np.float32).flatten()
        return state

    # returns accumulated rewards from sub-objectives
    def reward_info(self):
        print(self.total_reward_contents)
        return self.total_reward_contents

    # perform simulation step
    def step(self, action, is_skip=False):
        done = False
        self.iternation += 1

        # get current x positon and compute change
        position = self.robot_center.getPosition()[0]
        distance = position - self.last_position
        self.last_position = position
        
        # clip action if required, execute action afterwards
        action_vector = action
        action_vec, rescaled_action_vector = self._transform_action(action_vector)
        self._act(rescaled_action_vector)

        # read custom data field from robot node, finished if supervisor wrote 'reset'
        custom_data = self.robot.getCustomData()
        if custom_data == "reset":
            done = True
            self.robot.setCustomData('')

        # ignore, not longer relevant
        old_state = self.state
        self.state = self._build_state()
        if not is_skip:
            self.state_buffer.append(self.state)

        # create state for next timestep
        next_state = np.array(self.state_buffer, dtype=np.float32).flatten()

        # remember distance for plotting
        self.total_distance += distance

        # fall penalty
        # large negative reward if robot fell
        fall_penalty = 0
        if done:
            # robot always falls
            fall_penalty = 50

        # distance sub-objective
        distance_reward = (100 * distance) 

        # target distance objective, NOT used
        target_distance = 3.0
        target_distance_penalty = 0.1 * np.square(target_distance - distance_reward) 

        # alive bonus
        alive_reward = 0.75

        # motor movement penalty    
        movement = np.array(self.state)[-8:] - np.array(old_state)[-8:]
        
        movement_penalty = 2 * np.sum(np.square(movement))

        # action clip penalty
        clip_actions = action_vec - rescaled_action_vector
        clip_action_penalty = 0.05 * np.sum(np.square(clip_actions))

        # store values for plot
        self.total_reward_contents[0] += distance_reward
        self.total_reward_contents[1] += alive_reward
        self.total_reward_contents[2] -= movement_penalty
        self.total_reward_contents[3] -= clip_action_penalty
        self.total_reward_contents[4] -= target_distance_penalty

        # total reward is reward sum of all sub-objectives
        reward = distance_reward + alive_reward - fall_penalty - movement_penalty - clip_action_penalty - target_distance_penalty
        # make reward smaller, so that MSE critic updates do not get to large
        reward = reward / 20.0

        return next_state, reward, done

