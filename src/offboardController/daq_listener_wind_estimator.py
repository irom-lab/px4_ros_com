import lcm
from exlcm import voltages_t

# For real-time wind estimation
from sensor_to_wind import NeuralNetworkSpeed
import torch
import numpy as np

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

# (From previous test) load zero wind
avg_nowind = np.array([2.77999738, 2.72527673, 2.72070981, 2.66512991, 2.80024984]).reshape((1,5))

# From example
def my_handler(channel, data):
    msg = voltages_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   voltages    = %s" % str(msg.voltages))
    numpy_input = np.array([msg.voltages])
    numpy_input = (numpy_input-avg_nowind) # subtract 0 vel avg
    numpy_input = np.abs(numpy_input) # take absolute value
    numpy_input = -np.sort(-numpy_input)[0,0:3] # sort in descending order
    print("   numpy input    = ",numpy_input, np.shape(numpy_input))
    torch_input = torch.from_numpy(numpy_input).reshape((1,3)).float()
    print("   torch input    = ",torch_input)
    wind_estimate = model_speed.forward(torch_input).item()
    print("   wind_estimate    = ",wind_estimate)
    print("")


# do this once
lc = lcm.LCM()
subscription = lc.subscribe("VOLTAGES", my_handler)

# do this frequently
try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass

# do this once
lc.unsubscribe(subscription)
