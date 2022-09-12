import lcm

from exlcm import flowdrone_t

lc = lcm.LCM()

msg = flowdrone_t()
msg.drone_state = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
msg.thrust_sp = 20.0
msg.body_rate_sp = [21.0, 22.0, 23.0]
msg.wind_magnitude_estimate = 24.0
msg.wind_angle_estimate = 25.0
msg.thrust_residual = 26.0
msg.body_rate_residual = [27.0, 28.0, 29.0]

lc.publish("FLOWDRONE", msg.encode())