import lcm

from exlcm import voltages_t

# From example
def my_handler(channel, data):
    msg = voltages_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   voltages    = %s" % str(msg.voltages))
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
        lc.handle()
except KeyboardInterrupt:
    pass

# do this once
lc.unsubscribe(subscription)
