import lcm

from exlcm import flowdrone_t

def my_handler(channel, data):
    msg = flowdrone_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   drone_state    = %s" % str(msg.drone_state))
    print("   thrust_sp = %s" % str(msg.thrust_sp))
    print("   thrust_residual = %s" % str(msg.thrust_residual))
    print("   body_rate_sp =%s" % str(msg.body_rate_sp))
    print("   body_rate_residual =%s" % str(msg.body_rate_residual))
    print("   wind_magnitude_estimate        = %s" % str(msg.wind_magnitude_estimate))
    print("   wind_angle_estimate     = %s" % str(msg.wind_angle_estimate))
    print("")

lc = lcm.LCM()
subscription = lc.subscribe("FLOWDRONE", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass