import lcm

from exlcm import voltages_t

# for debugging/attempt
import time
import select
from collections import namedtuple
import numpy as np

latest_msg = namedtuple("latest_msg", ["voltages", "timestamp"])
latest_msg.voltages = np.zeros(5)
latest_msg.timestamp = 0.0

# From example
def my_handler(channel, data):
    msg = voltages_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   voltages    = %s" % str(msg.voltages))
    print(time.time())
    global latest_msg 
    latest_msg = msg
    print("")

# def my_handler(channel, data):
#     msg = voltages_t.decode(data)
#     # print("Received message on channel \"%s\"" % channel)
#     return msg.timestamp, msg.voltages

# do this once
lc = lcm.LCM()
subscription = lc.subscribe("VOLTAGES", my_handler)


# do this frequently
try:
    while True:
        #lc.handle() #example default

 ########## attempt from https://github.com/lcm-proj/lcm/blob/master/examples/python/listener_select.py

        timeout = 0  # amount of time to wait, in seconds
        while True:
            rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
            if rfds:
                lc.handle()
            else:
                print(latest_msg.voltages)
                # print("Waiting for message...")        timeout = 0
        

except KeyboardInterrupt:
    pass

# do this once
lc.unsubscribe(subscription)

# # This example demonstrates how to use LCM with the Python select module

# import select
# import lcm
# from exlcm import example_t
# from collections import namedtuple


# def my_handler(channel, data):
#     msg = example_t.decode(data)
#     print("Received message on channel \"%s\"" % channel)
#     print("   timestamp   = %s" % str(msg.timestamp))
#     print("   position    = %s" % str(msg.position))
#     print("   orientation = %s" % str(msg.orientation))
#     print("   ranges: %s" % str(msg.ranges))
#     print("   name        = '%s'" % msg.name)
#     print("   enabled     = %s" % str(msg.enabled))
#     print("")
#     global latest_msg 
#     latest_msg = msg

# lc = lcm.LCM()
# lc.subscribe("EXAMPLE", my_handler)

# latest_msg = namedtuple("latest_msg", "position")
# latest_msg.position = 0

# try:
#     timeout = 0 # 1.5  # amount of time to wait, in seconds
#     while True:
#         rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
#         if rfds:
#             lc.handle()
            
#         else:
#             print(latest_msg.position)
#             # print("Waiting for message...")
# except KeyboardInterrupt:
#     pass