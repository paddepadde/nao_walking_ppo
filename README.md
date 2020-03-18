# Learning to Walk with Proximate Policy Optimization

This repository contains the code of a robotics project if did for a university course. 
The end-goal was to teach the NAO robot to walk by directly controling the robot joint angles in a reinforcement learning setting. 

The robot is trained by using the an Actor-Critic style architecture in combination with the Proximate Policy Optimization (PPO) [(Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) reinforcement learning algorithm. The neural networks and learning procedure is implemented with Tensorflow. 

The robot is never directly told or shown how walking works. He has to discover this notion exclussivly by trail-and-error learning. 

## Simulation Setup

To simulate the NAO robot model, the [Webots Robot Simulator](https://github.com/cyberbotics/webots) is used.

Two controllers are used to setup the simulation. A robot controller manages the robot, by reading the sensor values, controlling the robot by setting the angles, and computing the reward.   
A supervisor controller is used to manage the simulation. This controller resets the simulation when a fall of the robot is detected.

The environment setup is kept simple so that high simulation speed can be guaranteed during the simulation:

<img src="https://imgur.com/8ty6NKE.png" width="500">

Unfortunaty, the Webots Reset Function reloads the controllers and all saved variables every time an episode is ended, therfore also potentually reseting the tensorflow model. I solved this issue by offloading the tensorflow functionaliy (training and gettting actions) to a seperate script that functions as a type of server. 
The robot controllers then communicate with this server via TCP sockets. 

## How does It Work?

An episode in the simulation repeats the same four steps in a loop and collects the observed data for the training:

* __Build State__: Build the state vector _s_ the robot is in. The vector contains the episode timestep, sensor values from the integrated gyroscope, accelerometer, position values, as well as robot joint. 
* __Act__: Get the angle values _a_ for the current state of the robot. This depends on the learned policy _π(a|s)_. 
* __Compute Reward__: Compute the reward _r_ that the robot receives for the action. Tested reward components conatined positive reward for moving forward on the x-axis, not falling over, as well as negative reward for falling over, excessive joint movement, and setting unreachable angles. 
* __Episode End Detection__: End the episode if the robot y-position crosses a fixed threshold or the number of timesteps extended a fixed threshold. 

After a certain number of simulation samples are collected, the actor-critic neural networks controlling the robot are trained with the collected samples. See the included PPO paper at the start of the README for details on the training procedure. 


## Learned Walking Control Policies

Due to the high sample-complexity of the PPO algorithm and the high computational effort of the simulator the training of the robot could take up to several days. I trained the robot with reward functions of different complexity and scaling. These animations show some of the learned control policies: 

__Animations of the Results:__

<img src="https://i.imgur.com/pDDrpUt.gif" width="300"> <img src="https://i.imgur.com/bhKBZWf.gif" width="300">

<img src="https://i.imgur.com/QojydTD.gif" width="300"> <img src="https://i.imgur.com/xUft4MM.gif" width="300">

Most of the learned control policies are definitly close to a style human walking. Because real walking is never shown to the Robot, he developes weird walking styles in some experiments (2nd animation) or unnatural walking characteristics (e.g. excessive arm movement).  


## Project Structure 

* `/controllers/ppo_supervisor` Webots controller for simulation management. 
* `/controllers/ppo_walking` Webots robot walking controller (state, actions, reward computation). 
* `/nao_walking` Webots file for world

## Instructions 

Instructions for setting up the project are included in the `INSTRUCTIONS.md` file in the root directory of the repository.  
