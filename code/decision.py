
import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None and len(Rover.sample_angles) == 0:
        if Rover.vel == 0 and not Rover.throttle == 0:
            if len(Rover.nav_angles) >= Rover.stop_forward:
                Rover.steer = -15
            else:
                Rover.brake = 0
                Rover.throttle = -Rover.throttle_set
                Rover.mode = 'forward'
                Rover.steer = -15
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                angle_count = len(Rover.nav_angles)
                turn_angle = Rover.nav_angles[round(2*angle_count/3)]
                Rover.steer = np.clip((turn_angle * 180/np.pi), -15, 15)
                #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    angle_count = len(Rover.nav_angles)
                    turn_angle = Rover.nav_angles[round(2*angle_count/3)]
                    Rover.steer = np.clip((turn_angle * 180/np.pi), -15, 15)
                    #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    elif len(Rover.sample_angles) > 0 and not Rover.near_sample:
        print('Sample angle and no near sample')
        Rover.steer = np.clip(np.mean(Rover.sample_angles * 180/np.pi), -15, 15)
        Rover.throttle = Rover.throttle_set
        Rover.brake = 0
        Rover.mode = 'forward'
        #if len(Rover.nav_angles) < Rover.stop_forward:
            #Rover.steer = -15
            #Rover.throttle = Rover.throttle_set
            #Rover.brake = 0
            #Rover.mode = 'forward'
    elif Rover.near_sample:
        print('Near sample')
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = 'stop'
        # Commented to stop the Rover from picking up sample
        #Rover.send_pickup = True
        Rover.near_sample = False
        Rover.picking_up = False
        Rover.steer = -15
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    return Rover
