import numpy as np
from robot_sensory import OracleBasedOdometrySensor, ProximityCollisionSensor, ConstrainedVisualSensor, LateralisedVisualSensor
from utils import radian_absolute_difference, radian_difference_a2b, queue
from mushroom_body_new import MultiOutputMushroomBody


class MBbase:
    def __init__(self,
                 simulator,
                 robot,
                 memory_capacity=6, # 6 or 10 at full speed, robot hits wall after 7 or 8 epochs looking through it
                 N_kc=10000,
                 N_pn_perkc=10,
                 S_kc=0.01,
                 N_mbon=2,
                 preprocess='di',   # h decorrelates figures more than di
                 sigma12=(1,2)):
        self.eye = ConstrainedVisualSensor(simulator,
                                           robot,
                                           preprocess=preprocess,
                                           sigma12=sigma12)
        self.MB = MultiOutputMushroomBody(N_pn=self.eye.view_size,
                                          N_kc=N_kc,
                                          N_mbon=N_mbon,
                                          N_pn_perkc=N_pn_perkc,
                                          S_kc=S_kc,
                                          kc2mbon_rule='decay')
        # this threshold is dependent on visual preprocessing
        self.memocap = memory_capacity
        self.reset()

    def reset(self):
        self.vl = queue(self.memocap)
        self.va = queue(self.memocap)
        self.kc = queue(self.memocap)
        self.mbon = queue(self.memocap)

    def update(self):
        pn = self.eye.get_view().flatten()
        kc = self.MB.hashing(pn)[0]
        mbon = self.MB.evaluating(kc)[0]
        self.kc.update(kc)
        self.mbon.update(mbon)

    def find_orn(self, orn_head, orn_PI):
        return orn_PI

    def learn(self, *args):
        pass



class MB4ON(MBbase):
    def __init__(self,
                 simulator,
                 robot,
                 memory_capacity=6, # 6 or 10 at full speed, robot hits wall after 7 or 8 epochs looking through it
                 N_kc=10000,
                 N_pn_perkc=10,
                 S_kc=0.01,
                 preprocess='di',   # h decorrelates figures more than di
                 sigma12=(1,2),
                 lesion=[]):
        super().__init__(simulator=simulator,
                         robot=robot,
                         memory_capacity=memory_capacity,
                         N_kc=N_kc,
                         N_pn_perkc=N_pn_perkc,
                         S_kc=S_kc,
                         N_mbon=4,
                         preprocess=preprocess,
                         sigma12=sigma12)
        self.lesion_multiplier = np.ones(4)
        if len(lesion) > 0:
            self.lesion_multiplier[lesion] = 0

    def mbon2control(self):
        return 0, 0

    def find_orn(self, orn_head, orn_PI):
        confidence_forward, rotation = self.mbon2control()
        orn_MB = orn_head + np.pi * int(confidence_forward < 0)
        orn_diff = radian_difference_a2b(orn_PI, orn_MB)
        orn_CX = orn_PI + orn_diff * np.abs(confidence_forward)
        orn = orn_CX + rotation * np.sign(np.pi / 2 - radian_absolute_difference(orn_head, orn_CX))
        return orn

    def _get_mbon(self):
        return self.mbon[-1] * self.lesion_multiplier

    def _learn_rate(self, d_fblr):
        return d_fblr

    def learn(self, d_fblr):
        mbon_rates = np.multiply(self._learn_rate(d_fblr), self.lesion_multiplier)
        self.MB.learning(self.kc[0], mbon_rates)


class MB4ONaxial(MB4ON):
    def mbon2control(self):
        mbon_forward, mbon_backward, mbon_left, mbon_right = self._get_mbon()
        rotation = (mbon_left - mbon_right) * np.pi #/ 2
        confidence_forward = mbon_forward - mbon_backward
        return confidence_forward, rotation

    def _learn_rate(self, d_fblr):
        # mbon[0] - activates forward movement;
        # mbon[1] - backward
        # mbon[2] - left turn;
        # mbon[3] - right turn
        return d_fblr


class MB4ONdiagonal(MB4ON):
    def mbon2control(self):
        mbon_fl, mbon_fr, mbon_bl, mbon_br = self._get_mbon()
        rotation = (mbon_fl + mbon_bl - mbon_fr - mbon_br) * np.pi / 4
        confidence_forward = (mbon_fl + mbon_fr - mbon_bl - mbon_br) / 2
        return confidence_forward, rotation

    def _learn_rate(self, d_fblr):
        # mbon[0] - activates forward left movement;
        # mbon[1] - forward right
        # mbon[2] - backwoard left;
        # mbon[3] - backward right
        d_f, d_b, d_l, d_r = d_fblr
        d_fl = d_f + d_l
        d_fr = d_f + d_r
        d_bl = d_b + d_l
        d_br = d_b + d_r
        return d_fl, d_fr, d_bl, d_fr


class Goal_follower:
    def __init__(self,
                 odometry_sensor:OracleBasedOdometrySensor,
                 obstacle_sensor: ProximityCollisionSensor,
                 goal_position,
                 steer_bias=[0,0]):
        self.odometry_sensor = odometry_sensor
        self.obstacle_sensor = obstacle_sensor
        self.pos_goal = np.array(goal_position)
        self.init_dist2goal = np.linalg.norm(self.pos_goal - self.odometry_sensor.sense_position())
        self.vl_bias, self.va_bias = steer_bias

        self.steer_mode = 'default'
        self.reset()

    def reset(self):
        self.dist2goal = queue(2, self.init_dist2goal)
        self.orn2goal = queue(2, self._orn2pos(self.pos_goal))

    def _orn2pos(self, pos_target):
        pos = self.odometry_sensor.sense_position()
        vec2target = pos_target - pos
        orn2target = np.arctan2(vec2target[1], vec2target[0])
        return orn2target

    def _orn2target_bymode(self, vec_force):
        force = np.linalg.norm(vec_force)
        orn2target = self._orn2pos(self.pos_goal)
        return orn2target

    def _steer2targetorn(self, orn2target):
        orn = self.odometry_sensor.sense_orientation()
        self.correction_rad = radian_difference_a2b(orn, orn2target)
        vl = np.cos(self.correction_rad)
        va = -np.sin(self.correction_rad)
        return vl + self.vl_bias, va + self.va_bias

    def _update_body_status(self):
        dist2goal = np.linalg.norm(self.pos_goal - self.odometry_sensor.sense_position())
        self.dist2goal.update(dist2goal)
        orn2goal = self._orn2pos(self.pos_goal)
        self.orn2goal.update(orn2goal)
        self.vec_force = self.obstacle_sensor.feel_force(magnitude_only=False)
        self.sense_vlva = self.odometry_sensor.sense_speed_linear(), self.odometry_sensor.sense_speed_angular()

    def steer(self):
        self._update_body_status()
        orn2target = self._orn2target_bymode(self.vec_force)
        vl, va = self._steer2targetorn(orn2target)
        return vl, va

class MBCX_driver:
    def __init__(self,
                 odometry_sensor:OracleBasedOdometrySensor,
                 obstacle_sensor:ProximityCollisionSensor,
                 MB:MB4ON,
                 goal_position,
                 steer_bias=[0,0]):
        self.odometry_sensor = odometry_sensor
        self.obstacle_sensor = obstacle_sensor
        self.MB = MB
        self.pos_goal = np.array(goal_position)
        self.init_dist2goal = np.linalg.norm(self.pos_goal - self.odometry_sensor.sense_position())

        self.duration_obstacle_avoidance = 15
        self.vl_bias, self.va_bias = steer_bias

        self.reset()

    def reset(self):
        self.obstacle_avoidance_goal = self.pos_goal
        self.countdown_obstacle_avoidance = 0
        self.MB.reset()
        self.steer_mode = queue(2, 'MB-CX')
        self.dist2goal = queue(2, self.init_dist2goal)
        self.orn2goal = queue(2, self._orn2pos(self.pos_goal))
        self.orn_escape = self.orn2goal[-1]

    def _orn2pos(self, pos_target):
        pos = self.odometry_sensor.sense_position()
        vec2target = pos_target - pos
        orn2target = np.arctan2(vec2target[1], vec2target[0])
        return orn2target

    def _orn2target_bymode(self, vec_force):
        force = np.linalg.norm(vec_force)
        if force < self.obstacle_sensor.thre_force and self.countdown_obstacle_avoidance > 0:
            self.countdown_obstacle_avoidance -= 1
            self.steer_mode.update('reflex')
            orn_diff = radian_difference_a2b(self.orn_escape_end, self.orn_escape)
            orn2target = self.orn_escape_end + orn_diff * self.countdown_obstacle_avoidance / self.duration_obstacle_avoidance
        else:
            orn_head = self.odometry_sensor.sense_orientation()
            if force >= self.obstacle_sensor.thre_force:
                self.countdown_obstacle_avoidance = self.duration_obstacle_avoidance
                self.steer_mode.update('reflex')
                self.orn_escape = np.arctan2(vec_force[1], vec_force[0])
                self.orn_escape_end = np.arctan2(*vec_force)
                if radian_absolute_difference(self.orn_escape_end, orn_head) > np.pi / 2:
                    self.orn_escape_end += np.pi
                orn2target = self.orn_escape
                print('      collision!')
            else:
                self.steer_mode.update('MB-CX')
                orn2target = self.MB.find_orn(orn_head, self.orn2goal[-1])
        return orn2target

    def _steer2targetorn(self, orn2target):
        orn = self.odometry_sensor.sense_orientation()
        self.correction_rad = radian_difference_a2b(orn, orn2target)
        vl = np.cos(self.correction_rad)
        va = -np.sin(self.correction_rad)
        return vl + self.vl_bias, va + self.va_bias

    def _update_body_status(self):
        dist2goal = np.linalg.norm(self.pos_goal - self.odometry_sensor.sense_position())
        self.dist2goal.update(dist2goal)
        orn2goal = self._orn2pos(self.pos_goal)
        self.orn2goal.update(orn2goal)
        self.vec_force = self.obstacle_sensor.feel_force(magnitude_only=False)
        self.sense_vlva = self.odometry_sensor.sense_speed_linear(), self.odometry_sensor.sense_speed_angular()

    def _depression_from_reinforcement(self):
        escape_on = self.steer_mode[-2] != 'reflex' and self.steer_mode[-1] == 'reflex'
        escape_off = self.steer_mode[-2] == 'reflex' and self.steer_mode[-1] != 'reflex'
        target_front = self.sense_vlva[0] > 0
        target_back = self.sense_vlva[0] < 0
        target_left = self.correction_rad < 0
        target_right = self.correction_rad > 0
        d_f, d_b, d_l, d_r = np.zeros(4)
        if escape_on or escape_off:
            d_f = self._d_rate((escape_on and target_front) or (escape_off and target_back), 1)
            d_b = self._d_rate((escape_on and target_back) or (escape_off and target_front), 1)
            d_l = self._d_rate((escape_on and target_left) or (escape_off and target_right), 1)
            d_r = self._d_rate((escape_on and target_right) or (escape_off and target_left), 1)
        # else:
        #     distance_gain = (self.dist2goal[-2] - self.dist2goal[-1]) / self.init_dist2goal
        #     rotation_gain = radian_difference_a2b(self.orn2goal[-1], self.orn2goal[-2]) / np.pi
        #     move2goal_reward = distance_gain * self.sense_vlva[0]
        #     head2goal_reward = rotation_gain * self.sense_vlva[1]
        #
        #     d_f = self._d_rate((distance_gain < 0 and target_front) or (distance_gain > 0 and target_back), move2goal_reward)
        #     d_b = self._d_rate((distance_gain < 0 and target_back) or (distance_gain > 0 and target_front), move2goal_reward)
        #     d_l = self._d_rate((rotation_gain < 0 and target_left) or (rotation_gain > 0 and target_right), head2goal_reward)
        #     d_r = self._d_rate((rotation_gain < 0 and target_right) or (rotation_gain > 0 and target_left), head2goal_reward)
        return d_f, d_b, d_l, d_r

    def _d_rate(self, condition, reward):
        return int(condition) * np.abs(reward)

    def steer(self):
        self._update_body_status()
        self.MB.update()
        orn2target = self._orn2target_bymode(self.vec_force)
        vl, va = self._steer2targetorn(orn2target)
        d_fblr = self._depression_from_reinforcement()
        self.MB.learn(d_fblr)
        return vl, va

    def trial_end_learn(self, goal_reached):
        # this is not necessarily good
        goal_missed = not goal_reached
        target_front = self.sense_vlva[0] > 0
        target_back = self.sense_vlva[0] < 0
        target_left = self.correction_rad < 0
        target_right = self.correction_rad > 0
        d_f = self._d_rate((goal_missed and target_front) or (goal_reached and target_back), 1)
        d_b = self._d_rate((goal_missed and target_back) or (goal_reached and target_front), 1)
        d_l = self._d_rate((goal_missed and target_left) or (goal_reached and target_right), 1)
        d_r = self._d_rate((goal_missed and target_right) or (goal_reached and target_left), 1)
        self.MB.learn([d_f, d_b, d_l, d_r])

        # homeostatic process may only be necessary if each locomotion step a model learns
        # mbw = self.MB.MB.kc2mbon.W_hash2val
        # print('mb weights', np.mean(mbw), np.median(mbw))
        # more dramatically: if goal_missed, change PN-KC for KC-MBON == 1