import lcm

from exlcm import voltages_t

# for debugging/attempt
import time
import select

# From example
def my_handler(channel, data):
    msg = voltages_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   voltages    = %s" % str(msg.voltages))
    print(time.time())
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

        # timeout = 1.5  # amount of time to wait, in seconds
        # while True:
        #     rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
        #     if rfds:
        #         lc.handle()
        #     else:
        #         print("Waiting for message...")
        timeout = 0
        lc.handle_timeout(timeout)
        

except KeyboardInterrupt:
    pass

# do this once
lc.unsubscribe(subscription)
