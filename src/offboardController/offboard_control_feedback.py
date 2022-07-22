# /****************************************************************************
#  *
#  * Copyright 2020 PX4 Development Team. All rights reserved.
#  *
#  * Redistribution and use in source and binary forms, with or without
#  * modification, are permitted provided that the following conditions are met:
#  *
#  * 1. Redistributions of source code must retain the above copyright notice, this
#  * list of conditions and the following disclaimer.
#  *
#  * 2. Redistributions in binary form must reproduce the above copyright notice,
#  * this list of conditions and the following disclaimer in the documentation
#  * and/or other materials provided with the distribution.
#  *
#  * 3. Neither the name of the copyright holder nor the names of its contributors
#  * may be used to endorse or promote products derived from this software without
#  * specific prior written permission.
#  *
#  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  * POSSIBILITY OF SUCH DAMAGE.
#  *
#  ****************************************************************************/

# /**
#  * @brief Offboard control with state estimation from PX4 and controller from gym-pybullet-drones PX4
#  * @file offboard_control_feedback.py
#  * @ author < nsimon@princeton.edu >
#  * @ author Nate Simon
#  * @ based on: offboard_control_py.py
#  * @ author < hanghaotian@gmail.com >
#  * @ author Haotian Hang
#  */

# ROS2 Imports
import rclpy
from rclpy.node import Node
import sys
# Math Imports
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

# PX4 Imported Messages - Offboard Mode
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import Timesync
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleControlMode
# PX4 Imported Messages - State Estimation
# from px4_msgs.msg import VehicleAttitude
# from px4_msgs.msg import VehicleLocalPosition
# from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import VehicleOdometry
# PX4 Imported Messages - Control
from px4_msgs.msg import VehicleRatesSetpoint

# PX4Control Imports
from PX4Control import PX4Control
from envs.BaseAviary import DroneModel

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_control')
        # initialize parameters
        self.offboard_setpoint_counter_ = 0  # counter for the number of setpoints sent
        self.timestamp_ = 0  # in python because of GIL,  the basic operations are already atomic

        # there must be a better way to create placeholder initial values - are these necessary?
        self.vehicle_attitude_ = 0
        self.vehicle_local_position_ = 0
        self.vehicle_global_position_ = 0

        self.offboard_control_mode_publisher_ = self.create_publisher(
            OffboardControlMode, "fmu/offboard_control_mode/in", 10)
        self.trajectory_setpoint_publisher_ = self.create_publisher(
            TrajectorySetpoint, "fmu/trajectory_setpoint/in", 10)
        self.vehicle_rates_setpoint_publisher_ = self.create_publisher(
            VehicleRatesSetpoint, "fmu/vehicle_rates_setpoint/in", 10)
        self.vehicle_command_publisher_ = self.create_publisher(
            VehicleCommand, "fmu/vehicle_command/in", 10)
        self.timesync_sub_ = self.create_subscription(
            Timesync, "fmu/timesync/out", self.timesync_callback, 10)

        # self.vehicle_attitude_sub_ = self.create_subscription(
        #     VehicleAttitude, "fmu/vehicle_attitude/out", self.vehicle_attitude_callback, 10)
        # self.vehicle_local_position_sub_ = self.create_subscription(
        #     VehicleLocalPosition, "fmu/vehicle_local_position/out", self.vehicle_local_position_callback, 10)
        # self.vehicle_global_position_sub_ = self.create_subscription(
        #     VehicleGlobalPosition, "fmu/vehicle_global_position/out", self.vehicle_global_position_callback, 10)
        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry, "fmu/vehicle_odometry/out", self.vehicle_odometry_callback, 10)

        timer_period = 0.1  # seconds
        self.timer_ = self.create_timer(
            timer_period, self.timer_callback)

    def timesync_callback(self, msg):
        self.timestamp_ = msg.timestamp
    def vehicle_odometry_callback(self, msg):
        #TO-DO: Check orientations
        self.vehicle_pos_ = np.array([msg.x,msg.y,msg.z])
        self.vehicle_quat_ = msg.q
        self.vehicle_rpy_ = p.getEulerFromQuaternion(self.vehicle_quat_) # convert from quat or VehicleAttitudeSetpoint ?
        self.vehicle_vel_ = np.array([msg.vx,msg.vy,msg.vz])
        self.vehicle_ang_v_ = np.array([msg.rollspeed,msg.pitchspeed,msg.yawspeed])

    # def vehicle_attitude_callback(self, msg):
    #     self.vehicle_quat_ = msg.q
    #     #        self.vehicle_rpy_ = np.array([msg.x,msg.y,msg.z]) # convert from quat or VehicleAttitudeSetpoint ?
    # def vehicle_local_position_callback(self, msg):
    #     self.vehicle_pos_ = np.array([msg.x,msg.y,msg.z])
    #     self.vehicle_vel_ = np.array([msg.vx,msg.vy,msg.vz])
    #     #self.vehicle_local_position_state_ = np.array([msg.x,msg.y,msg.z])
    #     # self.vehicle_local_position_state_ = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
    #     #                    self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
    # def vehicle_global_position_callback(self, msg):
    #     self.vehicle_global_position_ = msg.lat

    def timer_callback(self):
        if (self.offboard_setpoint_counter_ == 10):
            # Change to Offboard mode after 10 setpoints
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, float(1), float(6))
            self.arm()
        # offboard_control_mode needs to be paired with trajectory_setpoint
        self.publish_offboard_control_mode()
        #self.publish_trajectory_setpoint()
        # addition of vehicle_rates_setpoint - presumed paired with offboard_control_mode from previous lines
        self.publish_vehicle_rates_setpoint()

        if (self.offboard_setpoint_counter_ < 11):
            self.offboard_setpoint_counter_ += 1

    # @brief  Send a command to Arm the vehicle

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    # @brief  Send a command to Disarm the vehicle
    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("Disarm command send")

    # @brief Publish the offboard control mode. For this example, only position and altitude controls are active.
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp_
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        #self.get_logger().info("offboard control mode publisher send")
        self.offboard_control_mode_publisher_.publish(msg)

    # @ brief Publish a trajectory setpoint For this example, it sends a trajectory setpoint to make the vehicle hover at 5 meters with a yaw angle of 180 degrees.
    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp_
        x = 3
        y = 2
        z = -15
        msg.position = np.array([np.float32(x), np.float32(y), np.float32(z)])
        # self.get_logger().info("trajectory setpoint send")
        self.trajectory_setpoint_publisher_.publish(msg)

    def get_state_from_pixhawk(self):
        quat = self.vehicle_rpy_
        print('quat: ' + str(quat) + ', shape: ' + str(np.shape(quat)))


    def get_body_rates_from_PX4_Control(self):
        ctrl = [
            PX4Control(drone_model=DroneModel.X500, Ts=1e-2)
        ]
        np.set_printoptions(precision=2)
        # print(str(self.vehicle_quat_))
        # rpy_degs = np.asarray(self.vehicle_rpy_)*180/np.pi
        # print(rpy_degs.astype('int'))
        # rpy_degs_comparison = R.from_quat([self.vehicle_quat_]).as_euler('zyx', degrees=True)
        # print(rpy_degs_comparison.astype('int'))
        state_vec = np.hstack([self.vehicle_pos_, self.vehicle_quat_, self.vehicle_rpy_,
                           self.vehicle_vel_, self.vehicle_ang_v_, np.zeros(4)])
        # print('state_vec: ' + str(state_vec) + ', shape: ' + str(np.shape(state_vec)))
        # print('\r' + str(state_vec[0:3]), end='')
        # print('control: ')
        control = ctrl[0].computeRateAndThrustFromState(
            state=state_vec, #proper shape: (20,)
            target_pos=np.array([0,0,-2.5]), #proper shape: (3,)
        )
        return control

    # @ brief Publish a vehicl rates setpointsource ~/px4_ros_com_ros2/install/setup.bash
    def publish_vehicle_rates_setpoint(self):
        control = self.get_body_rates_from_PX4_Control()
        # it is evident that the thrust from control is not normalized
        max_thrust = 2*9.81*1.2 #roughly 2:1 thrust/weight ratio
        #self.get_state_from_pixhawk()
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.timestamp_
        #print(self.timestamp_)
        #works in jmavsim but not in gazebo?
        msg.roll = -control[0][0]/180*np.pi # minus sign in jmavsim due to orientation issues
        msg.pitch = control[0][1]/180*np.pi
        msg.yaw = control[0][2]/180*np.pi
        msg.thrust_body = np.array([np.float32(0.0), np.float32(0.0), np.float32(-control[1]/max_thrust)])
        np.set_printoptions(precision=3)
        print('rates: ' + str(np.array([msg.roll, msg.pitch, msg.yaw])))
        print('thrust: ' + str(msg.thrust_body[2]))
        self.vehicle_rates_setpoint_publisher_.publish(msg)
    #  @ brief Publish vehicle commands
    #  @ param command   Command code(matches VehicleCommand and MAVLink MAV_CMD codes)
    #  @ param param1    Command parameter 1
    #  @ param param2    Command parameter 2

    def publish_vehicle_command(self, command, param1, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp_
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher_.publish(msg)


def main(argc, argv):
    print("Starting offboard control node...")
    rclpy.init()
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    rclpy.shutdown()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
