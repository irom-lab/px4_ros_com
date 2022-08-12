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
#  * @brief Offboard control with state estimation from PX4 and controller architecture from gym-pybullet-drones and Quadcopter_SimCon (for PX4-Autopilot in python)
#  * @built from: https://github.com/utiasDSL/gym-pybullet-drones / https://github.com/bobzwik/Quadcopter_SimCon/blob/master/Simulation/ctrl.py
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
from px4_msgs.msg import VehicleOdometry
# PX4 Imported Messages - Control
from px4_msgs.msg import VehicleRatesSetpoint

# PX4Control Imports
from PX4Control import PX4Control
from envs.BaseAviary import DroneModel # from gym-pybullet-drones framework
from utils.control import RotToQuat, quatMultiply, quatInverse, quat2Dcm

# Imports for debugging
import time
import matplotlib.pyplot as plt
# Debugging variables. ctrl+F "ie " / "ie_" to identify such variables... loggie, countie, etc...
countie = 0
duration = 25
countie_end = duration*10 # 10 works for a 10 Hz sampling rate
loggie = np.zeros((countie_end,10))
print('loggie shape: '+ str(np.shape(loggie)))

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_control')
        # initialize parameters
        self.offboard_setpoint_counter_ = 0  # counter for the number of setpoints sent
        self.timestamp_ = 0  # in python because of GIL,  the basic operations are already atomic

        # TODO: better way to create placeholder values - are these necessary?
        self.vehicle_pos_ = 0
        self.vehicle_quat_ = 0
        self.vehicle_rpy_ = 0
        self.vehicle_vel_ = 0
        self.vehicle_ang_v_ = 0
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

        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry, "fmu/vehicle_odometry/out", self.vehicle_odometry_callback, 10)

        timer_period = 0.1  # seconds
        self.timer_ = self.create_timer(
            timer_period, self.timer_callback)

    def timesync_callback(self, msg):
        self.timestamp_ = msg.timestamp
    def vehicle_odometry_callback(self, msg):
        # self.vehicle_pos_ = np.array([msg.position]).reshape((3,)) # see: version of vehicle_odometry
        self.vehicle_pos_ = np.array([msg.x,msg.y,msg.z]) #NED, meters
        self.vehicle_quat_ = msg.q #FRD to reference (q_offset is from reference to navigation?)
        self.vehicle_rpy_ = p.getEulerFromQuaternion(self.vehicle_quat_) # Euler angles not used for control
        self.vehicle_vel_ = np.array([msg.vx,msg.vy,msg.vz]) #NED, m/s
        # self.vehicle_vel_ = np.array([msg.velocity]).reshape((3,)) # see: version of vehicle_odometry
        self.vehicle_ang_v_ = np.array([msg.rollspeed,msg.pitchspeed,msg.yawspeed]) #FRD (body-fixed frame) rad/s
        # self.vehicle_ang_v_ = np.array([msg.angular_velocity]).reshape((3,)) # see: version of vehicle_odometry

    def timer_callback(self):
        if (self.offboard_setpoint_counter_ == 10):
            # Change to Offboard mode after 10 setpoints
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, float(1), float(6))
            self.arm()
        # offboard_control_mode needs to be paired with trajectory_setpoint
        self.publish_offboard_control_mode() # comment to print outputs but not switch mode autmomatically
        # publish setpoint here: (note: must match offboard_control_mode)
        # self.publish_trajectory_setpoint()
        self.publish_vehicle_rates_setpoint() # note: NED

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
        msg.x = 3.0
        msg.y = 2.0
        msg.z = -15.0
        # msg.position = np.array([np.float32(x), np.float32(y), np.float32(z)]) # Old definition
        # self.get_logger().info("trajectory setpoint send")
        self.trajectory_setpoint_publisher_.publish(msg)

    def get_body_rates_from_PX4_Control(self):
        ctrl = [
            PX4Control(drone_model=DroneModel.X500, Ts=1e-2)
        ]
        # np.set_printoptions(precision=2)

        # MATCH STATE WITH gym-pybullet-drones https://github.com/utiasDSL/gym-pybullet-drones#observation-spaces-examples
        # X, Y, Z position in WORLD_FRAME (in meters, 3 values)
        # Quaternion orientation in WORLD_FRAME (4 values)
        # Roll, pitch and yaw angles in WORLD_FRAME (in radians, 3 values)
        # The velocity vector in WORLD_FRAME (in m/s, 3 values)
        # Angular velocity in WORLD_FRAME (3 values)
        # Motors' speeds (in RPMs, 4 values)

        #NOTE: vehicle_ang_v_ is in body frame but has NO EFFECT on quad
        state_vec = np.hstack([self.vehicle_pos_, self.vehicle_quat_, self.vehicle_rpy_,
                           self.vehicle_vel_, self.vehicle_ang_v_, np.zeros(4)])
        # TODO: do motors' speeds (RPMs) = np.zeros(4) matter for PX4Control?
        control = ctrl[0].computeRateAndThrustFromState(
            state=state_vec, #proper shape: (20,)
            target_pos=np.array([0,0,-3.0]), #proper shape: (3,)
        )
        return control

    # @ brief Publish a vehicle rates setpoint
    def publish_vehicle_rates_setpoint(self):
        control = self.get_body_rates_from_PX4_Control()
        # it is evident that the thrust from PX4Control is not normalized
        # quick and dirty way to normalize using roughly 2:1 T/W ratio
        max_thrust = 2*9.81*1.2 # without 1.2 factor, drone does not have descent authority
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.timestamp_
        # BODY ANGULAR RATES IN NED FRAME (rad/sec)
        msg.roll = control[0][0]
        msg.pitch = control[0][1]
        msg.yaw = control[0][2]
        # thrust output of PX4Control must be normalized and negated
        msg.thrust_body = np.array([np.float32(0.0), np.float32(0.0), -np.float32(control[1]/max_thrust)])
        # Debugging variables (for plots)
        global countie,loggie

        loggie[countie,0:4] = np.array([self.vehicle_quat_])
        loggie[countie,4:7] = np.array([msg.roll,msg.pitch,msg.yaw])
        loggie[countie,7:10] = control[2]

        countie += 1
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
    # # Original code
    # print("Starting offboard control node...")
    # rclpy.init()
    # offboard_control = OffboardControl()
    # rclpy.spin(offboard_control)
    # rclpy.shutdown()
    # # Adapted code for debugging, plots after duration seconds
    print("Starting offboard control node...")
    rclpy.init()
    offboard_control = OffboardControl()
    start_time = time.time()
    while (time.time() - start_time) < duration:
        rclpy.spin_once(offboard_control, timeout_sec=0.1) 
    rclpy.shutdown()
    fig = plt.figure()
    ax2 = plt.subplot(2, 1, 1)
    ax2.plot(np.transpose(loggie[:,4]),label='x')
    ax2.plot(np.transpose(loggie[:,5]),label='y')
    ax2.plot(np.transpose(loggie[:,6]),label='z')
    ax2.legend()
    ax2.set_title('Body Rate Setpoint')
    ax2.set_ylabel('rad/s')
    ax2.set_xlabel('deciseconds')
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(np.transpose(loggie[:,7]),label='x')
    ax3.plot(np.transpose(loggie[:,8]),label='y')
    ax3.plot(np.transpose(loggie[:,9]),label='z')
    ax3.legend()
    ax3.set_title('Pos Error')
    ax3.set_ylabel('m')
    ax3.set_xlabel('deciseconds')

    plt.subplots_adjust(hspace=0.6)
    plt.show()
    #plt.savefig('/home/nate/Desktop/attitude_br.png')

    print('Goodbye, countie = '+str(countie))


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
