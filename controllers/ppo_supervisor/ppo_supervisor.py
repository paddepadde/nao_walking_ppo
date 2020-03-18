from controller import Supervisor, Emitter

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
print("Starting Supervisor...")

episode_step = 0

robot = supervisor.getFromDef('NAO')
translation_field = robot.getField('translation')
# rotation_field = robot.getField('rotation')
custom_data = robot.getField('customData')

initial_translation = [0, 0.35, 0]
initial_rotation = [-1, 0, 0, 1.57]

reset_robot = False

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1: 
    episode_step += 1

    robot_position = translation_field.getSFVec3f()

    y_position = robot_position[1]

    if reset_robot:
        supervisor.step(timestep)
        supervisor.simulationReset()
        #translation_field.setSFVec3f(initial_translation)
        #rotation_field.setSFRotation(initial_rotation)
        
    if y_position < 0.29 or episode_step > 512:
        custom_data.setSFString('reset')
        reset_robot = True
