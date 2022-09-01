import torch
from torch import nn
import time

#############################################################################
# Neural network speed estimation model

class NeuralNetworkSpeed(nn.Module):
    '''
    A general/generic Neural Network model class for use with Pytorch. 
    
    TODO: include layer widths, types, and nonlinearities as inputs and dynamically allocate
          --- this will allow for custom classes rather than the clunky "if" statement used here. 
    '''
    def __init__(self, crosswire=False, fullAngles=False, geom=6):
        super(NeuralNetworkSpeed, self).__init__()
        self.flatten = nn.Flatten()
        
        if crosswire:
            # Architecture to use for crosswire prediction
            # Input size is 3  --- three crosswire features
            # Output size is 2 --- speed (m/s) and angle (rad)
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(3, 25),
                nn.ReLU(),
                nn.Linear(25, 15),
                nn.ReLU(),
                nn.Linear(15, 2),
            )
        else:
            if fullAngles:
                # Architecture to use for angle prediction if the data is dense (2-degree increments)
                # Input size is 6  --- six sensor readings (voltages)
                # Output size is 1 --- angle (rad)
                k1 = int(geom*8)
                k2 = int(k1/2 + 5)
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(geom, k1),
                    nn.ReLU(),
                    nn.Linear(k1, k2),
                    nn.ReLU(),
                    nn.Linear(k2, 1),
                )
            else:
                # Architecture to use for speed prediction (generally) and for angle prediction 
                # if the data is NOT dense (e.g., is in 10-degree increments)
                # Input size is 6  --- six sensor readings (voltages)
                # Output size is 1 --- either speed (m/s) or angle (rad)
                '''
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(geom, 6),
                    nn.ReLU(),
                    nn.Linear(6, 3),
                    nn.ReLU(),
                    nn.Linear(3, 1),
                )
                '''
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(3, 6),
                    nn.ReLU(),
                    nn.Linear(6, 1),
                )

    def forward(self, x):
        # Method to propagate input (reading) through the network to get a prediction. 
        # Terminology is clunky because this is adapted from a classification example, hence 
        # the use of 'logits' even though we are doing regression.
        
        # TODO -- tidy up variable names, usage, etc (see above)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#############################################################################



#############################################################################
model_speed = NeuralNetworkSpeed(crosswire=False, fullAngles=False, geom=5)
opt = torch.optim.Adam(model_speed.parameters(), lr=0.001)
model_speed_path = "/home/ubuntu/companion-computer-clipboard/SavedModels/Velocity/N1_G5_Loocv5_best.tar"
checkpoint = torch.load(model_speed_path)

model_speed.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(model_speed.eval())


#############################################################################



#############################################################################
# Run model on test random input
input = torch.zeros((1,3))

t_start = time.time()
output = model_speed.forward(input)
t_end = time.time()

print("Model input: ", input)
print("Model output: ", output)
print("Inference time (s): ", t_end - t_start)


    