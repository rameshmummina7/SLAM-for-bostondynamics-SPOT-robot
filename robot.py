import logging
import time
import open3d as o3d

from project.robot_wrapper.robot_wrapper import SpotRobotWrapper, Velocity2D
import numpy as np

class Robot(SpotRobotWrapper):
    def _init_(self, config):
        super(Robot, self)._init_(config)

    def init_robot(self):
        # TODO: this part will be executed once during start up, any initialization should be done here
        if self.motors_on:
            self.stand_up()

        time.sleep(1)

        # v = Velocity2D(0,0,0)
        # self.velocity_command(v)

        logging.info("Robot initialized")
        #locationinitilised = 0
        #mapvariable = o3d.geometry.PointCloud()
        #print("robot.py")


    def loop_robot(self):
        ## TODO: this is the part where your code is executed repeatedly, you can use this to control the robot or retrieve data continuously from the robot



        if self.config.dbg_mode:
            # TODO: if you want to execute stuff only in the debug mode do it here
            self.auto_walk()