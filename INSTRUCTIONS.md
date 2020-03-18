# Instructions for running the project

## Used packages 

> Python Version 3.6.7  
> Webots R2019a Revision 1

Installed pip packages:

> tensorflow==1.12.0  
> numpy==1.16.2

## Starting Instructions 

Make sure Webots is closed.  
Navigate to robot controller.  

> cd {project_root}/controllers/ppo_walking

Execute agent with 

> python3 agent_server.py

Wait until program prints

> "Started Agent... Listening to Events"

Launch Webots to start the training. 
Do not close Webots while the episode is active, 
otherwise the agent_server socket will be stuck.   
World is stored in:

> {project_root}/nao_walking/walking.wbt

## Optional Information 

You may want to restart Webots every 24 hours, because the simulation speed slows down over time. 
Simply close and reopen Webots after an episode is complete. Do not close agent_server.py.
(Use Webots step function to search for end)

Informations are logged into `/log` folder. Display with:

> tensorboard --logdir=log

The name of the log file can be changed in the agent.py file: 

```python
    log_dir = './log/run_1'
    self.writer = tf.summary.FileWriter(log_dir)
```

Models are saved in `/models` folder. 

Training can continue from checkpoint if interrupted. (Sorry I was young and didn't know better...)
Reload by prodiving checkpoint argument to the agent in the agent_server.py and
setting correct step and episode:

```python

    agent = Agent(state_size=16, action_size=8, checkpoint='./model/walking__{x}')
    # ...
    step = 20153
    episode = 4120
    
```

Restart by starting server again

> python3 agent_server.py

## Project Folder Structure

`/controllers/ppo_supervisor` Webots controller for supervisor (check terminal and reset)    
`/controllers/ppo_walking` Webots robot walking controller (state, actions, reward)  
`/nao_walking` Webots file for world    

