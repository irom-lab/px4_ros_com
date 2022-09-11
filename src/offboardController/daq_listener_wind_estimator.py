# for DAQ listening
import lcm
from exlcm import voltages_t
from collections import namedtuple
import select

# For real-time wind estimation
from sensor_to_wind import NeuralNetworkSpeed, NeuralNetworkAngle
import torch
import numpy as np

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
    global voltage_sums
    global filter_counter
    global last5_array
    # if np.allclose(latest_msg.voltages, msg.voltages, rtol=1e-05):
    #     pass
    #     #print("No new voltages received")
    # else:
    if filter_N5:
        if filter_counter < 5:
            # print("np.shape(np.array(msg.voltages)[0:geometry]) shape: ",np.array(msg.voltages)[0:geometry])
            # print("np.array([msg.voltages])[0][0:geometry] shape: ",np.array([msg.voltages])[0][0:geometry])
            voltage_sums = voltage_sums + np.array([msg.voltages])[0][0:geometry]
            filter_counter = filter_counter + 1
            # print("filter_counter: ",filter_counter)
            # print("voltage_sums: ",voltage_sums)
        else:
            #numpy_input = np.array([msg.voltages])[0][0:geometry]
            numpy_input = voltage_sums/5
            filter_counter = 0 #reset
            voltage_sums = voltage_sums*0 #reset
            
            #print("avg voltages: ",numpy_input)
            numpy_input = (numpy_input-zero_wind_voltages[0]) # subtract 0 vel avg
            #print("zeroed voltages: ",numpy_input)
            numpy_input_speed = np.abs(numpy_input[0:geometry]) # take absolute value # to force adjacency: [2:geometry]
            numpy_input_speed = -np.sort(-numpy_input_speed)[0:3] # sort in descending order
            #print("4: ",numpy_input_speed)
            torch_input_speed = torch.from_numpy(numpy_input_speed).reshape((1,3)).float()
            torch_input_angle = torch.from_numpy(numpy_input).reshape((1,geometry)).float()
            #print("   torch input    = ",torch_input)
            wind_estimate = model_speed.forward(torch_input_speed).item()-zero_wind_estimate
            angle_estimate = model_angle.forward(torch_input_angle).item()*180/np.pi
            #print("   wind_estimate    = ",wind_estimate)

            # Update latest message
            latest_msg.voltages = msg.voltages
            latest_msg.wind_estimate = wind_estimate
            latest_msg.angle_estimate = angle_estimate
            print("voltages: ",latest_msg.voltages[0])
            print("wind_estimate: ",latest_msg.wind_estimate, "angle_estimate: ",latest_msg.angle_estimate)
            with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
                np.savetxt(f, np.array([time.time()-start_time, latest_msg.vx, latest_msg.vy, latest_msg.vz, latest_msg.wind_estimate, latest_msg.angle_estimate, numpy_input[0], numpy_input[1], numpy_input[2], numpy_input[3], numpy_input[4]]).reshape(1,11),delimiter=",", fmt='%1.5f')

    elif filter_movemax5:
        numpy_input = np.array([msg.voltages])[0][0:geometry]
        numpy_input = (numpy_input-zero_wind_voltages[0]) # subtract 0 vel avg
        #print("zeroed voltages: ",numpy_input)
        numpy_input_speed = np.abs(numpy_input[0:geometry]) # take absolute value # to force adjacency: [2:geometry]
        numpy_input_speed = -np.sort(-numpy_input_speed)[0:3] # sort in descending order
        #print("4: ",numpy_input_speed)
        torch_input_speed = torch.from_numpy(numpy_input_speed).reshape((1,3)).float()
        torch_input_angle = torch.from_numpy(numpy_input).reshape((1,geometry)).float()
        #print("   torch input    = ",torch_input)
        wind_estimate = model_speed.forward(torch_input_speed).item()-zero_wind_estimate
        angle_estimate = model_angle.forward(torch_input_angle).item()*180/np.pi
        #print("   wind_estimate    = ",wind_estimate)

        # insert another element to the filter
        last5_array[0:4] = last5_array[1:5]
        last5_array[4] = wind_estimate
        print("last5_array: ",last5_array)
        # Update latest message
        latest_msg.voltages = msg.voltages
        latest_msg.wind_estimate = max(last5_array)
        latest_msg.angle_estimate = angle_estimate
        print("wind_estimate: ",latest_msg.wind_estimate, "angle_estimate: ",latest_msg.angle_estimate)
        with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
            np.savetxt(f, np.array([time.time()-start_time, latest_msg.vx, latest_msg.vy, latest_msg.vz, latest_msg.wind_estimate, latest_msg.angle_estimate, numpy_input[0], numpy_input[1], numpy_input[2], numpy_input[3], numpy_input[4]]).reshape(1,11),delimiter=",", fmt='%1.5f')

    else:
        numpy_input = np.array([msg.voltages])[0][0:geometry]
        #print("1: ",numpy_input)
        numpy_input = (numpy_input-zero_wind_voltages[0]) # subtract 0 vel avg
        #print("2: ",numpy_input)
        numpy_input_speed = np.abs(numpy_input) # take absolute value
        #print("3: ",numpy_input_speed)
        numpy_input_speed = -np.sort(-numpy_input_speed)[0:3] # sort in descending order
        #print("4: ",numpy_input_speed)
        torch_input_speed = torch.from_numpy(numpy_input_speed).reshape((1,3)).float()
        torch_input_angle = torch.from_numpy(numpy_input).reshape((1,geometry)).float()
        #print("   torch input    = ",torch_input)
        wind_estimate = model_speed.forward(torch_input_speed).item()-zero_wind_estimate
        angle_estimate = model_angle.forward(torch_input_angle).item()*180/np.pi#-zero_wind_estimate
        #print("   wind_estimate    = ",wind_estimate)

        # Update latest message
        latest_msg.voltages = msg.voltages
        latest_msg.wind_estimate = wind_estimate
        latest_msg.angle_estimate = angle_estimate
        print("wind_estimate: ",latest_msg.wind_estimate, "angle_estimate: ",latest_msg.angle_estimate)
        with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
            np.savetxt(f, np.array([time.time()-start_time, latest_msg.vx, latest_msg.vy, latest_msg.vz, latest_msg.wind_estimate, latest_msg.angle_estimate, numpy_input[0], numpy_input[1], numpy_input[2], numpy_input[3], numpy_input[4]]).reshape(1,11),delimiter=",", fmt='%1.5f')

def main(args=None):
    subscription = lc.subscribe("VOLTAGES", my_handler)

    try:
        while True:
            lc.handle() #example default

    except KeyboardInterrupt:
        pass
    
    lc.unsubscribe(subscription)


if __name__ == "__main__":

    geometry = 5

    # Load the speed neural network
    model_speed = NeuralNetworkSpeed(crosswire=False, fullAngles=False, geom=geometry)
    opt = torch.optim.Adam(model_speed.parameters(), lr=0.001)
    model_speed_path = "/home/ubuntu/px4_ros_com_ros2/src/px4_ros_com/src/offboardController/models/N5_G"+str(geometry)+"_Loocv5_best.tar"
    checkpoint = torch.load(model_speed_path)
    model_speed.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model_speed.eval()

    model_angle = NeuralNetworkAngle(crosswire=False, fullAngles=True, geom=geometry)
    opt = torch.optim.Adam(model_angle.parameters(), lr=0.001)
    model_angle_path = "/home/ubuntu/px4_ros_com_ros2/src/px4_ros_com/src/offboardController/models/N5_G"+str(geometry)+"_best.tar"
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

    # used for filtering
    filter_N5 = False
    filter_counter = 0
    voltage_sums = np.zeros(5,)
    # used for movemax filtering
    filter_movemax5 = True
    last5_array = np.zeros([5,])
    
    if (filter_N5 and filter_movemax5):
        print("filter_N5 and filter_movemax5 cannot both be True")
        exit()

    # start time
    start_time = time.time()
                                    
    zero_wind_voltages = np.array([2.76386,      2.78803,      2.68601,      2.80839,      2.81071]).reshape((1,geometry)) # in air, hover, position mode
    zero_wind_estimate = model_speed.forward(torch.zeros(1,3)).item()
    print("zero_wind_estimate: ",zero_wind_estimate)

    with open("output.csv", "a") as f:   # use 'a' instead of 'ab'
        f.write("timestamp, vx, vy, vz, wind_estimate, angle_estimate, a0, a1, a2, a3, a4, a5")
        f.write("\n")

    main()
