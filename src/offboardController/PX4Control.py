import math
import numpy as np
from scipy.optimize import nnls

from control.BaseControl import BaseControl
from envs.BaseAviary import DroneModel
from utils.control import RotToQuat, quatMultiply, quatInverse, quat2Dcm


rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
radps2rpm = 30 / np.pi
rpm2radps = np.pi / 30


# Normalize quaternion, or any vector
def vectNormalize(q):
    return q / np.linalg.norm(q)


class PX4Control(BaseControl):
    """Based on https://github.com/bobzwik/Quadcopter_SimCon/blob/master/Simulation/ctrl.py#L278, which implements the PX4 controller in Python

    """

    ############################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 Ts: float = 1e-2):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.X500:
            print(
                "[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.X500"
            )
            exit()

        self.orient = 'ENU'  # z up
        self.g = 9.8

        # Set up params
        if self.DRONE_MODEL == DroneModel.X500:
            self.mB = 1.2105
            dxm = dym = 0.25
            MAX_RPM = 8000
            self.pos_P_gain = np.array([0.95, 0.95, 1.0])
            self.vel_P_gain = np.array([1.8, 1.8, 4.0])
            self.vel_D_gain = np.array([0.2, 0.2, 0.0])
            self.vel_I_gain = np.array([0.4, 0.4, 2])
            self.rate_P_gain = np.array([0.15, 0.15, 0.2])
            self.rate_D_gain = np.array([0.003, 0.003, 0.0])

        elif self.DRONE_MODEL == DroneModel.CF2X:
            self.mB = 0.0397
            MAX_RPM = 21000
            dxm = dym = 0.0397
            self.pos_P_gain = np.array([1.0, 1.0, 1.0])
            self.vel_P_gain = np.array([1.0, 1.0, 2.0])
            self.vel_D_gain = np.array([0.1, 0.1, 0.005])
            self.vel_I_gain = np.array([0.01, 0.01, 0.01])
            self.rate_P_gain = np.array([0.0003, 0.0003, 0])
            self.rate_D_gain = np.array([0.00001, 0.000001, 0])

        # Attitude P gains
        Pphi = 6.5  # 1.0
        Ptheta = Pphi
        Ppsi = 2.8  # 0.2
        self.att_P_gain = np.array([Pphi, Ptheta, Ppsi])

        self.maxThr = (4 * self.KF * MAX_RPM**2)  # 0.5953
        self.minThr = (1 * self.KF * MAX_RPM**2)  # ? 0.1488
        self.useIntergral = 1
        self.Ts = Ts

        self.tiltMax = 50.0 * deg2rad
        pMax = 220.0 * deg2rad
        qMax = 220.0 * deg2rad
        rMax = 200.0 * deg2rad
        self.rateMax = np.array([pMax, qMax, rMax])

        self.saturateVel_separetely = False
        self.velMax = np.array([5.0, 5.0, 5.0])
        self.velMaxAll = 5.0

        # Mixing (ENU orient)
        # kTh = self.KF * radps2rpm**2  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        # kTo = self.KM * radps2rpm**2  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        kTh = self.KF
        kTo = self.KM

        self.mixerFM = np.array(
            [[kTh, kTh, kTh, kTh],
             [
                 dym * kTh / np.sqrt(2), dym * kTh / np.sqrt(2),
                 -dym * kTh / np.sqrt(2), -dym * kTh / np.sqrt(2)
            ],
                [
                 -dxm * kTh / np.sqrt(2), dxm * kTh / np.sqrt(2),
                 dxm * kTh / np.sqrt(2), -dxm * kTh / np.sqrt(2)
            ], [-kTo, kTo, -kTo, kTo]]
        )  # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds

        self.mixerFMinv = np.linalg.inv(self.mixerFM)
        # self.minWmotor = 1000  # Minimum motor rotation speed (rad/s)
        # self.maxWmotor = np.sqrt(
        #     (2.25 * self.mB * self.g) /
        #     (4 * self.KF))  # Maximum motor rotation speed (rad/s)
        self.minWmotor = 2000  # Minimum motor rotation speed (rad/s)
        self.maxWmotor = MAX_RPM

        self.setYawWeight()
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.thr_int = np.zeros(3)
        self.vel = np.zeros(3)
        self.vel_dot = np.zeros(3)
        self.omega = np.zeros(3)
        self.omega_dot = np.zeros(3)

    ################################################################################

    def computeControlFromState(self,
                                state,
                                target_pos,
                                rate_residual=np.zeros((3)),
                                thrust_residual=0):
        """Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1

        # Get prev
        self.prev_vel = self.vel
        self.prev_omega = self.omega

        # States
        self.pos = state[0:3]
        self.quat = np.hstack(
            (state[6], state[3:6]))  # [qx,qy,qz,qw] -> [qw,qx,qy,qz]
        self.vel = state[10:13]
        self.ang_vel = state[13:16]  # in world frame

        self.dcm = quat2Dcm(self.quat)
        self.omega = np.dot(
            self.dcm.transpose(),
            self.ang_vel)  # body rate
        # print(self.omega)
        #
        self.vel_dot = (self.vel - self.prev_vel) / self.Ts
        self.omega_dot = (self.omega - self.prev_omega) / self.Ts
        # print('vel dot: ', self.vel_dot)
        # print('omega dot: ', self.omega_dot)

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp = target_pos
        self.vel_sp = np.zeros((3))
        self.acc_sp = np.zeros((3))
        self.thrust_sp = np.zeros((3))
        self.eul_sp = np.zeros((3))
        self.pqr_sp = 0
        self.yawFF = 0

        # Cascaded control
        self.pos_control()
        self.saturateVel()
        self.z_vel_control()
        self.xy_vel_control()
        self.thrustToAttitude()
        self.attitude_control()

        # Rate controller
        self.rate_sp += rate_residual
        self.rate_control()

        # Get thrust
        thrust = np.linalg.norm(self.thrust_sp) + thrust_residual

        # Mixing
        t = np.array(
            [thrust, self.rateCtrl[0],
             self.rateCtrl[1],
             self.rateCtrl[2]])
        # w_cmd = np.sqrt(
        #     np.clip(np.dot(self.mixerFMinv, t), self.minWmotor**2,
        #             self.maxWmotor**2))
        w_cmd = np.dot(self.mixerFMinv, t)
        if np.min(w_cmd) < 0:
            sol, res = nnls(self.mixerFM, t, maxiter=10)
            w_cmd = sol
        w_cmd = np.clip(w_cmd, self.minWmotor**2, self.maxWmotor**2)
        w_cmd = np.sqrt(w_cmd)

        # print('')
        # print('Pos: ', self.pos)
        # print('Rate sp:', self.rate_sp)
        # print('Thrust sp:', self.thrust_sp)
        # print('Input: ', t)
        # # print('Nominal: ', np.sqrt(np.dot(self.mixerFMinv, t)))
        # print('Output: ', w_cmd)
        # while 1:
        #     continue
        # print('')
        # import time
        # time.sleep(0.2)
        # return w_cmd * radps2rpm, None, None
        return w_cmd, None, None

    ################################################################################

    def computeRateAndThrustFromState(self,
                                      state,
                                      target_pos,
                                      rate_residual=np.zeros((3)),
                                      thrust_residual=0):
        """Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1

        # Get prev
        self.prev_vel = self.vel
        self.prev_omega = self.omega

        # States
        self.pos = state[0:3]
        self.quat = np.hstack(
            (state[6], state[3:6]))  # [qx,qy,qz,qw] -> [qw,qx,qy,qz]
        self.vel = state[10:13]
        self.ang_vel = state[13:16]  # in world frame

        self.dcm = quat2Dcm(self.quat)
        self.omega = np.dot(
            self.dcm.transpose(),
            self.ang_vel)  # body rate
        # print(self.omega)
        #
        self.vel_dot = (self.vel - self.prev_vel) / self.Ts
        self.omega_dot = (self.omega - self.prev_omega) / self.Ts
        # print('vel dot: ', self.vel_dot)
        # print('omega dot: ', self.omega_dot)

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp = target_pos
        self.vel_sp = np.zeros((3))
        self.acc_sp = np.zeros((3))
        self.thrust_sp = np.zeros((3))
        self.eul_sp = np.zeros((3))
        self.pqr_sp = 0
        self.yawFF = 0

        # Cascaded control
        self.pos_control()
        self.saturateVel()
        self.z_vel_control()
        self.xy_vel_control()
        self.thrustToAttitude()
        self.attitude_control()

        self.rate_sp += rate_residual
        thrust = np.linalg.norm(self.thrust_sp) + thrust_residual

        return self.rate_sp, thrust

    ################################################################################

    def pos_control(self):

        # Z Position Control
        # ---------------------------
        pos_error = self.pos_sp[0:3] - self.pos
        self.vel_sp[0:3] += self.pos_P_gain[0:3] * pos_error

    def saturateVel(self):

        # Saturate Velocity Setpoint
        # ---------------------------
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (self.saturateVel_separetely):
            self.vel_sp = np.clip(
                self.vel_sp, -self.velMax, self.velMax)
        else:
            totalVel_sp = np.linalg.norm(self.vel_sp)
            if (totalVel_sp > self.velMaxAll):
                self.vel_sp = self.vel_sp / totalVel_sp * self.velMaxAll

    def z_vel_control(self):

        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to
        # allow hover when the position and velocity error are nul
        vel_z_error = self.vel_sp[2] - self.vel[2]
        if (self.orient == "NED"):
            thrust_z_sp = self.vel_P_gain[2] * vel_z_error - self.vel_D_gain[
                2] * self.vel_dot[2] + self.mB * (self.acc_sp[2] -
                                                  self.g) + self.thr_int[2]
        elif (self.orient == "ENU"):
            thrust_z_sp = self.vel_P_gain[2] * vel_z_error - self.vel_D_gain[
                2] * self.vel_dot[2] + self.mB * (self.acc_sp[2] +
                                                  self.g) + self.thr_int[2]

        # Get thrust limits
        if (self.orient == "NED"):
            # The Thrust limits are negated and swapped due to NED-frame
            uMax = -self.minThr
            uMin = -self.maxThr
        elif (self.orient == "ENU"):
            uMax = self.maxThr
            uMin = self.minThr

        # Apply Anti-Windup in D-direction
        stop_int_D = (
            thrust_z_sp >= uMax and vel_z_error >= 0.0) or (
            thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += self.vel_I_gain[
                2] * vel_z_error * self.Ts * self.useIntergral
            # Limit thrust integral
            self.thr_int[2] = min(
                abs(self.thr_int[2]), self.maxThr) * np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self):

        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = self.vel_sp[0:2] - self.vel[0:2]
        thrust_xy_sp = self.vel_P_gain[0:2] * vel_xy_error - self.vel_D_gain[
            0:2] * self.vel_dot[0:2] + self.mB * (
                self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(
            self.thrust_sp[2]) * np.tan(self.tiltMax)
        thrust_max_xy = math.sqrt(
            self.maxThr**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy
                ** 2):
            mag = np.linalg.norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp / mag * thrust_max_xy
            print('saturate thrust')
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0 / self.vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp -
                                      self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += self.vel_I_gain[
            0:2] * vel_err_lim * self.Ts * self.useIntergral

    def thrustToAttitude(self):

        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -vectNormalize(self.thrust_sp)
        if (self.orient == "ENU"):
            body_z = -body_z

        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-math.sin(yaw_sp), math.cos(yaw_sp), 0.0])

        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = vectNormalize(body_x)

        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = RotToQuat(R_sp)

    def attitude_control(self):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = self.dcm[:, 2]
        e_z_d = -vectNormalize(self.thrust_sp)
        if (self.orient == "ENU"):
            e_z_d = -e_z_d

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + math.sqrt(
            np.linalg.norm(e_z)**2 * np.linalg.norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = vectNormalize(qe_red)

        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = quatMultiply(qe_red, self.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = quatMultiply(quatInverse(self.qd_red), self.qd_full)
        q_mix = q_mix * np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = quatMultiply(
            self.qd_red,
            np.array([
                math.cos(self.yaw_w * np.arccos(q_mix[0])), 0, 0,
                math.sin(self.yaw_w * np.arcsin(q_mix[3]))
            ]))

        # Resulting error quaternion
        self.qe = quatMultiply(quatInverse(self.quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0 * np.sign(self.qe[0]) *
                        self.qe[1:4]) * self.att_P_gain

        # Limit yawFF
        self.yawFF = np.clip(
            self.yawFF, -self.rateMax[2],
            self.rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_sp += quat2Dcm(quatInverse(self.quat)
                                 )[:, 2] * self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(
            self.rate_sp, -self.rateMax, self.rateMax)

    def rate_control(self):

        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - self.omega
        # Be sure it is right sign for the D part
        self.rateCtrl = self.rate_P_gain * rate_error - self.rate_D_gain * self.omega_dot

    def setYawWeight(self):

        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5 * (
            self.att_P_gain[0] + self.att_P_gain[1])
        self.yaw_w = np.clip(
            self.att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)
        self.att_P_gain[2] = roll_pitch_gain
