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

# THIS PARTICULAR VERSION WAS USED FOR TRAINING, TO LOG CERTAIN VALUES

# ROS2 Imports
import rclpy
from rclpy.node import Node
import sys
# Math Imports
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

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
from px4_msgs.msg import BatteryStatus

# PX4Control Imports
from PX4Control import PX4Control, DroneModel
from utils.control import RotToQuat, quatMultiply, quatInverse, quat2Dcm
from utils.mlp import MLP, normalize_obs

# LCM Imports
import lcm
from exlcm import flowdrone_t

# Imports for debugging
import time
import datetime
import matplotlib.pyplot as plt

duration = 45 # of entire run, sec

start_time = time.time()
frequency = 40 #Hz

ctrl = [
    PX4Control(drone_model=DroneModel.X500, Ts=1e-2)
]

lc = lcm.LCM()

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_control')
        # initialize parameters
        self.offboard_setpoint_counter_ = 0  # counter for the number of setpoints sent
        self.timestamp_ = 0  # in python because of GIL,  the basic operations are already atomic
        self.received_odometry = False

        # TODO: better way to create placeholder values - are these necessary?
        # NOTE: these zero initializations cause issues with the 'first setpoint message' (280 deg/sec yaw setpoint on 220902)

        self.offboard_position_mode = True # start in position mode, switch to body rates after (see next line)
        self.duration_position_sp = 15 #seconds
        
        self.vehicle_pos_ = np.array([0.0, 0.0, 0.0])
        self.vehicle_quat_ = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_rpy_ = np.array([0.0, 0.0, 0.0])
        self.vehicle_vel_ = np.array([0.0, 0.0, 0.0])
        self.vehicle_ang_v_ = np.array([0.0, 0.0, 0.0])
        self.voltage_v = 0.0
        self.voltage_filtered_v =  0.0
        self.current_a =  0.0
        self.current_filtered_a	 =  0.0
        self.initial_voltage_filtered_v = 0.0 # capture the value of voltage_filtered_v the first time that it is 

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
        # self.vehicle_rates_setpoint_sub_ = self.create_subscription(
        #     VehicleRatesSetpoint, "fmu/vehicle_rates_setpoint/out", self.vehicle_rates_setpoint_callback, 10)
        self.battery_status_sub_ = self.create_subscription(
            BatteryStatus, "/fmu/battery_status/out", self.battery_status_callback, 10)
        
        timer_period = 1/frequency  # seconds, 0.025 sec -> 40 Hz
        self.timer_ = self.create_timer(
            timer_period, self.timer_callback)

        # Load residual model
        # TODO: add cfg file
        # input_size = 15 # fix
        input_size = 12 # #unaware, trained in wind
        # layer_size_list = [input_size, 256, 256, 128, 128, 4] #unaware, trained without wind 
        layer_size_list = [input_size, 512, 256, 128, 128, 4] #unaware, trained in wind

        # layer_size_list = [input_size, 256, 256, 4] #old
        policy_path = 'models/step_rising_3_unaware.pth'
        self.mlp = MLP(dimList=layer_size_list,
                activation_type='relu',)
        self.mlp.load_weight(policy_path)
        # self.rate_residual_scale = 0.1 #fix
        self.rate_residual_scale = 0.3 #unaware, trained in wind
        self.thrust_residual_scale = 1


    def timesync_callback(self, msg):
        self.timestamp_ = msg.timestamp
    def vehicle_odometry_callback(self, msg):
        if not self.received_odometry:
            # first message
            self.received_odometry = True

        self.vehicle_pos_ = np.array([msg.x,msg.y,msg.z]) #NED, meters
        self.vehicle_quat_ = msg.q #FRD to reference (q_offset is from reference to navigation?)

        # TODO! Test
        rot = R.from_quat(self.vehicle_quat_)   # pybullet uses [x,y,z,w] for qusternion, scipy uses the same convention

        self.vehicle_rpy_  = rot.as_euler('ZYX', degrees=False) #ROLL, PITCH, then YAW -- EXTRINSIC
        self.vehicle_rpy_[1] = -self.vehicle_rpy_[1]            #This operation is required to match PX4 attitude
        self.vehicle_rpy_[2] = (self.vehicle_rpy_[2] % (2.0 * np.pi ) - np.pi) * -1.0     #This operation is required to match PX4 attitude

        # Roll, Pitch Safety Check
        if abs(self.vehicle_rpy_[0]) > 0.5235 or abs(self.vehicle_rpy_[1]) > 0.5235:
            print("Roll or pitch safety tolerance exceeded")
            exit()

        self.vehicle_vel_ = np.array([msg.vx,msg.vy,msg.vz]) #NED, m/s
        self.vehicle_ang_v_ = np.array([msg.rollspeed,msg.pitchspeed,msg.yawspeed]) #FRD (body-fixed frame) rad/s

    def battery_status_callback(self, msg):
        self.voltage_v = msg.voltage_v
        self.voltage_filtered_v = msg.voltage_filtered_v
        self.current_a = msg.current_a
        self.current_filtered_a	 = msg.current_filtered_a
        if self.voltage_filtered_v != 0.0 and self.initial_voltage_filtered_v == 0.0:
            print("First nonzero filtered voltage measured. Set voltage to ",self.voltage_filtered_v," for simple thrust mapping.")
            self.initial_voltage_filtered_v = self.voltage_filtered_v

    # def vehicle_rates_setpoint_callback(self, msg):
    #     # msg.roll
    #     # msg.pitch
    #     # msg.yaw
    #     print("body_thrust_sp: ",msg.thrust_body)


    def timer_callback(self):
        if (self.offboard_setpoint_counter_ == 10 and self.received_odometry):
            # Change to Offboard mode 10 setpoints AFTER receiving vehicle_odometry
            print("self.received_odometry",self.received_odometry)
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, float(1), float(6))
            #self.arm()

        # Publish offboard control mode 10 setpoints AFTER receiving vehicle_odometry
        if (self.offboard_setpoint_counter_ >= 10 and self.received_odometry):
            self.publish_offboard_control_mode() # comment to print outputs but not switch mode autmomatically
            
            # switch to body rate setpoints self.duration_position_sp seconds after offboard mode
            if ((time.time() - start_time) < self.duration_position_sp) and self.offboard_position_mode:
                #print("POSITION_SP")
                self.publish_trajectory_setpoint()
            else:
                self.offboard_position_mode = False
                #print("BODY_RATE_SP")
                self.publish_vehicle_rates_setpoint() # note: NED

        if (self.offboard_setpoint_counter_ < 11 and self.received_odometry):
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
        msg.position = self.offboard_position_mode
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = not self.offboard_position_mode
        print("msg.position: ",msg.position)
        print("msg.body_rate: ",msg.body_rate)

        #self.get_logger().info("offboard control mode publisher send")
        self.offboard_control_mode_publisher_.publish(msg)

    # @ brief Publish a trajectory setpoint For this example, it sends a trajectory setpoint to make the vehicle hover at 5 meters with a yaw angle of 180 degrees.
    def publish_trajectory_setpoint(self):
        # NOTE: the PX4 body frame is NED. The PX4 uses the compass to find NORTH and places the POSITIVE X direction in that direction.
        # that is, aligning the drone in the typical fashion (front pointed along the longer, longitudinal axis of the room) is about a 114-140 degree offset.
        yaw_offset = 1.91

        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp_
        msg.x = 0.0
        msg.y = 0.0
        msg.z = -1.0
        msg.yaw = yaw_offset
        print("position sp: ",np.array([np.float32(msg.x), np.float32(msg.y), np.float32(msg.z)]))
        # self.get_logger().info("trajectory setpoint send")
        self.trajectory_setpoint_publisher_.publish(msg)

    def get_body_rates_from_PX4_Control(self):

        # MATCH STATE WITH gym-pybullet-drones https://github.com/utiasDSL/gym-pybullet-drones#observation-spaces-examples
        # X, Y, Z position in WORLD_FRAME (in meters, 3 values)
        # Quaternion orientation in WORLD_FRAME (4 values)
        # Roll, pitch and yaw angles in WORLD_FRAME (in radians, 3 values)
        # The velocity vector in WORLD_FRAME (in m/s, 3 values)
        # Angular velocity in WORLD_FRAME (3 values)
        # Motors' speeds (in RPMs, 4 values)

        yaw_offset = 1.97 #140.0*np.pi/180 # [rad] what is the PX4's perceived yaw when oriented in the Forrestal Frame? (inspect ATTITUDE)

        # # setpoints in the Forrestal Frame
        # x_fr = 2.0         # [m]
        # y_fr = 3.0       # [m]
        # z_fr = -3.0        # [m]
        # yaw_fr = 0         # [rad] desired yaw in forrestal frame

        # pos_fr = np.array([[x_fr],[y_fr], [z_fr]])
        # #rotation matrix (about Z axis)
        # Rz = np.array([[np.cos(yaw_offset), -np.sin(yaw_offset), 0],
        #             [np.sin(yaw_offset),  np.cos(yaw_offset), 0],
        #             [0,               0,              1]])

        # # # rotate setpoints to NED 
        # x_NED,y_NED,z_NED = np.matmul(Rz,pos_fr)
        x_NED = 0.0
        y_NED = 0.0
        z_NED = -1.0
        #print(x_NED,y_NED,z_NED)
        #NOTE: vehicle_ang_v_ is in body frame but has NO EFFECT on quad
        state_vec = np.hstack([self.vehicle_pos_, self.vehicle_quat_, self.vehicle_rpy_,
                           self.vehicle_vel_, self.vehicle_ang_v_, np.zeros(4)])
        # TODO: do motors' speeds (RPMs) = np.zeros(4) matter for PX4Control?
        
        # Get residual
        if hasattr(self, 'mlp'):
            # switch from NED to RPY
            pos_ENU = np.array([self.vehicle_pos_[1],self.vehicle_pos_[0],-self.vehicle_pos_[2]])
            quat_gym_NED = np.hstack((self.vehicle_quat_[1:4],self.vehicle_quat_[0])) #back to [x,y,z,w] convention
            quat_gym_ENU = np.array([quat_gym_NED[1],quat_gym_NED[0],-quat_gym_NED[2],quat_gym_NED[3]])
            rpy_ENU = R.from_quat(quat_gym_ENU).as_euler('xyz', degrees=False)
            vel_ENU = np.array([self.vehicle_vel_[1],self.vehicle_vel_[0],-self.vehicle_vel_[2]])
            ang_vel_ENU = np.array([self.vehicle_ang_v_[1],self.vehicle_ang_v_[0],-self.vehicle_ang_v_[2]])
            obs_state = np.hstack((pos_ENU, rpy_ENU, vel_ENU, ang_vel_ENU))
            #print(obs_state)
            # obs_wind = np.zeros((3))    # TODO: add real wind observation
            obs_state = normalize_obs(obs_state)    # TODO: normalize wind
            # obs = torch.from_numpy(np.hstack((obs_state, obs_wind))).float()
            obs = torch.from_numpy(obs_state).float()
            residual = self.mlp.infer(obs)
        else:
            residual = np.zeros((4))

        # Un-normalize residual
        rate_residual = residual[:-1] * self.rate_residual_scale #*0#-0.3->0.3 rad/sec
        #increase pitch scale
        #rate_residual[1] *= 2
        thrust_residual = residual[-1] * self.thrust_residual_scale #*0#-1->1 N
        print("ENU residual: ", rate_residual, thrust_residual) # ENU
        NED_rate_residual = np.array([rate_residual[1],rate_residual[0],-rate_residual[2]])
        print("NED_rate_residual: ", NED_rate_residual) # NED

        # NOTE: the PX4 body frame is NED. The PX4 uses the compass to find NORTH and places the POSITIVE X direction in that direction.
        # that is, aligning the drone in the typical fashion (front pointed along the longer, longitudinal axis of the room) is about a 114-140 degree offset.
        control = ctrl[0].computeRateAndThrustFromState(
            state=state_vec, #proper shape: (20,)
            target_pos=np.array([float(x_NED),float(y_NED),float(z_NED)]), #proper shape: (3,),
            rate_residual=NED_rate_residual,
            thrust_residual=thrust_residual,
            target_rpy=np.array([0,0,yaw_offset]),
        )

        msg = flowdrone_t()
        msg.drone_state = state_vec
        msg.thrust_sp = control[1]
        msg.body_rate_sp = control[0]
        msg.wind_magnitude_estimate = 24.0
        msg.wind_angle_estimate = 25.0
        msg.thrust_residual = thrust_residual
        msg.body_rate_residual = NED_rate_residual

        lc.publish("FLOWDRONE", msg.encode())

        return control

    # @ brief Publish a vehicle rates setpoint
    def publish_vehicle_rates_setpoint(self):
        control = self.get_body_rates_from_PX4_Control()
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.timestamp_
        # BODY ANGULAR RATES IN NED FRAME (rad/sec)
        msg.roll = np.clip(control[0][0],-0.8727,0.8727) # clip roll rate to +-50 deg/sec
        msg.pitch = np.clip(control[0][1],-0.8727,0.8727) # clip pitch rate to +-50 deg/sec
        msg.yaw = np.clip(control[0][2],-0.174533,0.174533) # clip yaw rate setpoint to +-10 deg/sec
        #print("body rate, thrust: ", msg.roll, msg.pitch, msg.yaw, control[1])
        #print("voltage_filtere: ", self.voltage_filtered_v)
        thrust_clipped = np.clip(control[1], 0.0,0.7)
        msg.thrust_body = np.array([np.float32(0.0), np.float32(0.0), -np.float32(thrust_clipped)])
        # thrust output of PX4Control must be normalized and negated
        #msg.thrust_body = np.array([np.float32(0.0), np.float32(0.0), -np.float32(control[1]/max_thrust)])
        # Debugging variables (for plots)

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
    while (time.time() - start_time) < duration:
        rclpy.spin_once(offboard_control, timeout_sec=0.1) 
    rclpy.shutdown()
    print("Shutting down ROS node")

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
