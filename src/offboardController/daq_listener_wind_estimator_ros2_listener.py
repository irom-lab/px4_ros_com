# for DAQ listening
import lcm
from exlcm import voltages_t
from collections import namedtuple
import select

# For real-time wind estimation
from sensor_to_wind import NeuralNetworkSpeed
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
        numpy_input = np.array([msg.voltages])
        numpy_input = (numpy_input-avg_nowind) # subtract 0 vel avg
        numpy_input = np.abs(numpy_input) # take absolute value
        numpy_input = -np.sort(-numpy_input)[0,0:3] # sort in descending order
        torch_input = torch.from_numpy(numpy_input).reshape((1,3)).float()
        #print("   torch input    = ",torch_input)
        wind_estimate = model_speed.forward(torch_input).item()
        #print("   wind_estimate    = ",wind_estimate)
        #print("")

        # Update latest message
        latest_msg.voltages = msg.voltages
        latest_msg.wind_estimate = wind_estimate
    # print("Latest message:")
    # print(latest_msg.voltages, latest_msg.wind_estimate)


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
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("RECEIVED VEHICLE ODOMETRY DATA")
        print("=============================")
        print("ts: ", msg.timestamp)
        print("vx (NORTH): ", msg.vx)
        print("vy (EAST): ", msg.vy)
        print("vz (DOWN): ", msg.vz)
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
        print("Last received message:")
        print(latest_msg.voltages, latest_msg.wind_estimate)

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

    # Load the neural network
    model_speed = NeuralNetworkSpeed(crosswire=False, fullAngles=False, geom=5)
    opt = torch.optim.Adam(model_speed.parameters(), lr=0.001)
    model_speed_path = "/home/ubuntu/companion-computer-clipboard/SavedModels/Velocity/N1_G5_Loocv5_best.tar"
    checkpoint = torch.load(model_speed_path)
    model_speed.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model_speed.eval()

    # Initialize lcm
    lc = lcm.LCM()

    # first_timestep = 0
    latest_msg = namedtuple("latest_msg", ["voltages", "timestamp","wind_estimate"])
    latest_msg.voltages = np.zeros(5)
    latest_msg.wind_estimate = 0.0
    latest_msg.timestamp = 0.0

    # let's try to log data with a global (eek?) variable
    # timestamp, vx, vy, vz, wind_estimate
    # data_array = np.zeros((1,5))
    # with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
    #     f.write("timestamp, vx, vy, vz, wind_estimate")
    #     f.write("\n")

    # (From previous test) load zero wind
    # avg_nowind = np.array([2.80984, 2.83405, 2.80834, 2.68080, 2.79023]).reshape((1,5))
    #avg_nowind_GROUND_PROPS = np.array([2.70118513665594,	2.65209241961415,	2.7055819051447,	2.61474475080385,	2.72762385852091]).reshape((1,5))
    avg_nowind = np.array([2.709428235,	2.677134986,	2.672011659,	2.583482353,	2.723417954]).reshape((1,5)) # in air, hover, position mode

    main()
