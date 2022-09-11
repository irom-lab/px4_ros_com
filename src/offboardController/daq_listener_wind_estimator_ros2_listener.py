# for DAQ listening
import lcm
from exlcm import voltages_t
from collections import namedtuple
import select

# For real-time wind estimation
from sensor_to_wind import NeuralNetworkSpeed, NeuralNetworkAngle
import torch
import numpy as np

# for ROS2 listening
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
import threading

# for writing to file
import csv

# for debugging
import time


# From example
def my_handler(channel, data):
    msg = voltages_t.decode(data)
    # print("Received message on channel \"%s\"" % channel)
    # print("   timestamp   = %s" % str(msg.timestamp))
    # print("   voltages    = %s" % str(msg.voltages))

    global latest_msg

    if np.allclose(latest_msg.voltages, msg.voltages, rtol=1e-05):
        print("No new voltages received")
    else:
        numpy_input = np.array([msg.voltages])[0][0:geometry]
        print(numpy_input)
        numpy_input = (numpy_input-zero_wind_voltages) # subtract 0 vel avg
        numpy_input_speed = np.abs(numpy_input) # take absolute value
        numpy_input_speed = -np.sort(-numpy_input)[0,0:3] # sort in descending order
        torch_input_speed = torch.from_numpy(numpy_input_speed).reshape((1,3)).float()
        torch_input_angle = torch.from_numpy(numpy_input).reshape((1,geometry)).float()
        #print("   torch input    = ",torch_input)
        wind_estimate = model_speed.forward(torch_input_speed).item()#-zero_wind_estimate
        angle_estimate = model_angle.forward(torch_input_angle).item()*180/np.pi#-zero_wind_estimate
        #print("   wind_estimate    = ",wind_estimate)

        # Update latest message
        latest_msg.voltages = msg.voltages
        latest_msg.wind_estimate = wind_estimate
        latest_msg.angle_estimate = angle_estimate
        print("wind_estimate: ",latest_msg.wind_estimate, "angle_estimate: ",latest_msg.angle_estimate)
    # with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
    #     #print("array:",np.array([latest_msg.timestamp, latest_msg.vx, latest_msg.vy, latest_msg.vz, latest_msg.wind_estimate]))
    #     np.savetxt(f, np.array([latest_msg.timestamp, latest_msg.vx, latest_msg.vy, latest_msg.vz, latest_msg.wind_estimate]).reshape(1,5),delimiter=",")

class VehicleOdometryListener(Node):
    def __init__(self):
        super().__init__('vehicle_odometry_listener')
        # timer_period = 0.1  # seconds
        # self.timer_ = self.create_timer(
        #     timer_period, self.timer_callback)
        self.subscription = self.create_subscription(
            VehicleOdometry,
            "fmu/vehicle_odometry/out",
            self.listener_callback,
            1)
        self.subscription

    def listener_callback(self, msg):
        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        # print("RECEIVED VEHICLE ODOMETRY DATA")
        # print("=============================")
        # print("ts: ", msg.timestamp)
        # print("vx (NORTH): ", msg.vx)
        # print("vy (EAST): ", msg.vy)
        # print("vz (DOWN): ", msg.vz)
        # global data_array
        # global first_timestep
        # if first_timestep == 0:
        #     first_timestep = msg.timestamp
        # data_array[0,0:4] = np.array([(msg.timestamp-first_timestep)/1E6, msg.vx, msg.vy, msg.vz])
        ## TRY LCM
        timeout = 0.0  # amount of time to wait, in seconds
        #while True:
        rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
        if rfds:
            lc.handle()
        # else:
        #     print("No new message. Last received:")
        #     print(latest_msg.voltages, latest_msg.wind_estimate)

        global latest_msg

        latest_msg.vx = msg.vx
        latest_msg.vy = msg.vy
        latest_msg.vz = msg.vz
        latest_msg.timestamp = msg.timestamp

def main(args=None):
    # runs once
    subscription = lc.subscribe("VOLTAGES", my_handler)
    rclpy.init(args=args)
    print("Starting vehicle_odometry listener node...")
    vehicle_odometry_listener = VehicleOdometryListener()

    rclpy.spin(vehicle_odometry_listener) 
    # # Spin in a separate thread
    # thread = threading.Thread(target=rclpy.spin, args=(vehicle_odometry_listener, ), daemon=True)
    # thread.start()

    # # Seems to get stuck in loop
    # rate = vehicle_odometry_listener.create_rate(5.) #hz
    # while rclpy.ok():
    #     rate.sleep()

    vehicle_odometry_listener.destroy_node()
    rclpy.shutdown()
    lc.unsubscribe(subscription)


if __name__ == "__main__":

    geometry = 5

    # Load the speed neural network
    model_speed = NeuralNetworkSpeed(crosswire=False, fullAngles=False, geom=geometry)
    opt = torch.optim.Adam(model_speed.parameters(), lr=0.001)
    #model_speed_path = "/home/ubuntu/px4_ros_com_ros2/src/px4_ros_com/src/offboardController/models/N1_G5_Loocv4_best.tar"
    model_speed_path = "/home/ubuntu/px4_ros_com_ros2/src/px4_ros_com/src/offboardController/models/N1_G"+str(geometry)+"_Loocv5_best.tar"
    checkpoint = torch.load(model_speed_path)
    model_speed.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model_speed.eval()

    model_angle = NeuralNetworkAngle(crosswire=False, fullAngles=True, geom=geometry)
    opt = torch.optim.Adam(model_angle.parameters(), lr=0.001)
    model_angle_path = "/home/ubuntu/px4_ros_com_ros2/src/px4_ros_com/src/offboardController/models/N1_G"+str(geometry)+"_best.tar"
    checkpoint = torch.load(model_angle_path)

    model_angle.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model_angle.eval()

    # Initialize lcm
    lc = lcm.LCM()

    # first_timestep = 0
    latest_msg = namedtuple("latest_msg", ["voltages", "timestamp","wind_estimate","angle_estimate", "vx", "vy", "vz"])
    latest_msg.voltages = np.zeros(6)
    latest_msg.wind_estimate = 0.0
    latest_msg.angle_estimate = 0.0
    latest_msg.timestamp = 0.0
    latest_msg.vx = 0.0
    latest_msg.vy = 0.0
    latest_msg.vz = 0.0

    zero_wind_voltages = np.array([2.75875,     2.74712,     2.74717,     2.80839,     2.77488]).reshape((1,geometry)) # in air, hover, position mode
    zero_wind_estimate = model_speed.forward(torch.zeros(1,3)).item()
    print("zero_wind_estimate: ",zero_wind_estimate)
    #exit()
    # let's try to log data with a global (eek?) variable
    # timestamp, vx, vy, vz, wind_estimate
    # data_array = np.zeros((1,5))
    with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
        f.write("timestamp, vx, vy, vz, wind_estimate, angle_estimate")
        f.write("\n")

    # (From previous test) load zero wind
    # avg_nowind = np.array([2.80984, 2.83405, 2.80834, 2.68080, 2.79023]).reshape((1,5))
    #avg_nowind_GROUND_PROPS = np.array([2.70118513665594,	2.65209241961415,	2.7055819051447,	2.61474475080385,	2.72762385852091]).reshape((1,5))

    main()
