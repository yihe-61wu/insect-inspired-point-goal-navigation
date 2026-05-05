import numpy as np
from robot_sensory import OracleBasedOdometrySensor, ProximityCollisionSensor, ConstrainedVisualSensor, LateralisedVisualSensor
from utils import radian_absolute_difference, queue
from mushroom_body_new import MultiOutputMushroomBody


class DOMBDriver:
    def __init__(self,
                 simulator,
                 robot,
                 signal_threshold,
                 memory_capacity=6, # 6 or 10 at full speed, robot hits wall after 7 or 8 epochs looking through it
                 N_kc=10000,
                 N_pn_perkc=10,
                 S_kc=0.01,
                 preprocess='di',   # h decorrelates figures more than di
                 sigma12=(1,2)):
        self.eye = ConstrainedVisualSensor(simulator,
                                           robot,
                                           preprocess=preprocess,
                                           sigma12=sigma12)
        self.MB = MultiOutputMushroomBody(N_pn=self.eye.view_size,
                                          N_kc=N_kc,
                                          N_mbon=2,
                                          N_pn_perkc=N_pn_perkc,
                                          S_kc=S_kc,
                                          kc2mbon_rule='decay')
        self.signal_thre = signal_threshold
        # this threshold is dependent on visual preprocessing
        self.va_queue = queue(memory_capacity)
        self.kc_queue = queue(memory_capacity)
        self.mbon_queue = queue(memory_capacity)
        self.MBtakecontrol_queue = queue(memory_capacity)

    def update_kc(self):
        pn = self.eye.get_view().flatten()
        kc = self.MB.hashing(pn)
        self.kc_hist = self.kc_queue.update(kc)

    # these learning leads to tiny weights
    def learn_attractive(self):
        self.MB.learning(self.kc_hist[-1], [1, 0])

    def learn_repulsive(self):
        self.MB.learning(self.kc_hist[0], [0, 1])

    def evaluate(self):
        mbon = self.MB.evaluating(self.kc_hist[-1])[0]
        print('nov', mbon[0] / mbon[1])
        MBtakecontrol = (np.min(mbon) * self.signal_thre) < np.max(mbon)
        self.mbon_hist = self.mbon_queue.update(mbon)
        self.MBtakecontrol_hist = self.MBtakecontrol_queue.update(MBtakecontrol)

    def steer(self):
        mbon = self.mbon_hist[-1]
        if mbon[1] * self.signal_thre < mbon[0]:
            vl = -0.01
            if self.MBtakecontrol_hist[-2]:
                # va = np.sign(self.eye.oracle.get_speed_angular()) * 0.1
                va = self.va_hist[-1]
            else:
                va = -np.sign(self.eye.oracle.get_speed_angular()) * 0.1
        elif mbon[0] * self.signal_thre < mbon[1]:
            vl, va = 1, 0
        self.va_hist = self.va_queue.update(va)
        return vl, va


class RotateMBDriver:
    def __init__(self,
                 simulator,
                 robot,
                 signal_threshold,
                 memory_capacity=6, # 6 or 10 at full speed, robot hits wall after 7 or 8 epochs looking through it
                 N_kc=10000,
                 N_pn_perkc=10,
                 S_kc=0.01,
                 preprocess='di',   # h decorrelates figures more than di
                 sigma12=(1,2)):
        self.eye = ConstrainedVisualSensor(simulator,
                                           robot,
                                           preprocess=preprocess,
                                           sigma12=sigma12)
        self.MB = MultiOutputMushroomBody(N_pn=self.eye.view_size,
                                          N_kc=N_kc,
                                          N_mbon=4,
                                          N_pn_perkc=N_pn_perkc,
                                          S_kc=S_kc,
                                          kc2mbon_rule='decay')
        self.signal_thre = signal_threshold
        # this threshold is dependent on visual preprocessing
        self.va_queue = queue(memory_capacity)
        self.va_queue.memory = [0]
        self.kc_queue = queue(memory_capacity)
        self.mbon_queue = queue(memory_capacity)
        self.MBtakecontrol_queue = queue(memory_capacity)

    def reset(self):
        pass

    def update_kc(self):
        pn = self.eye.get_view().flatten()
        kc = self.MB.hashing(pn)
        self.kc_hist = self.kc_queue.update(kc)

    # these learning leads to tiny weights
    def learn_attractive(self):
        turning_right = int(self.va_queue[-1] >= 0)
        self.MB.learning(self.kc_hist[-1], [1 - turning_right, turning_right, 0, 0])

    def learn_repulsive(self):
        turning_right = int(self.va_queue[0] >= 0)
        self.MB.learning(self.kc_hist[0], [0, 0, 1 - turning_right, turning_right])

    def learn_route(self):
        # self.MB.learning(self.kc_hist[-1], self.mbon_hist[-1])
        pass


    def evaluate(self):
        mbon = self.MB.evaluating(self.kc_hist[-1])[0]
        print('nov', (mbon[0] + mbon[1]) / (mbon[2] + mbon[3]))
        MBtakecontrol = (np.minimum(mbon[0] + mbon[1], mbon[2] + mbon[3]) * self.signal_thre) < np.maximum(mbon[0] + mbon[1], mbon[2] + mbon[3])
        self.mbon_hist = self.mbon_queue.update(mbon)
        self.MBtakecontrol_hist = self.MBtakecontrol_queue.update(MBtakecontrol)

    def steer(self):
        mbon = self.mbon_hist[-1]
        if (mbon[3] + mbon[2]) * self.signal_thre < (mbon[1] + mbon[0]):
            vl, va = -0.01, np.sign(mbon[3] - mbon[2])
        elif (mbon[1] + mbon[0]) * self.signal_thre < (mbon[3] + mbon[2]):
            vl, va = 1, np.sign(mbon[0] - mbon[1])
        self.va_list = self.va_queue.update(va)
        return vl, va


class goal_directed_driver:
    def __init__(self,
                 odometry_sensor:OracleBasedOdometrySensor,
                 obstacle_sensor:ProximityCollisionSensor,
                 MB_driver:DOMBDriver,
                 robot_radius,
                 goal_position):
        self.odometry_sensor = odometry_sensor
        self.obstacle_sensor = obstacle_sensor
        self.MB_driver = MB_driver
        self.pos_goal = np.array(goal_position)

        self.duration_obstacle_avoidance = 15
        self.dist_escape = robot_radius * 2 * 100

        self.reset()

    def reset(self):
        self.obstacle_avoidance_goal = self.pos_goal
        self.steering_mode_curr = 'MB'
        self.steering_mode_prev = 'MB'
        self.countdown_obstacle_avoidance = 0

    def steer(self):
        self.steering_mode_prev = self.steering_mode_curr
        self.MB_driver.update_kc()
        vec_force = self.obstacle_sensor.feel_force(magnitude_only=False)
        force = np.linalg.norm(vec_force)
        if force < self.obstacle_sensor.thre_force and self.countdown_obstacle_avoidance > 0:
            self.countdown_obstacle_avoidance -= 1
            vl, va = self.steer2escape()
        else:
            self.MB_driver.evaluate()
            if force >= self.obstacle_sensor.thre_force:
                self.countdown_obstacle_avoidance = self.duration_obstacle_avoidance
                self.steering_mode_curr = 'reflex'
                vec_escape = vec_force / force
                # orn_force = np.arctan2(vec_force[1], vec_force[0])
                # orn_isright2force = (orn_force - self.odometry_sensor.sense_orientation()) % (2 * np.pi) < np.pi
                # orn_escape = orn_force + (np.pi / 2 * int(orn_isright2force) * 2 - 1)
                # orn_reflect = orn_force * 2 - self.odometry_sensor.sense_orientation() - np.pi
                # vec_escape = np.array([np.cos(orn_reflect), np.sin(orn_reflect)])
                # orn_surface = -np.arctan2(*vec_force)
                # orn_escape = orn_surface * 2 - self.odometry_sensor.sense_orientation()
                # vec_escape = np.array([np.cos(orn_escape), np.sin(orn_escape)])
                self.obstacle_avoidance_goal = self.odometry_sensor.sense_position() + vec_escape * self.dist_escape
                vl, va = self.steer2escape()
                print('      collision!')
            elif self.MB_driver.MBtakecontrol_hist[-1]:
                self.steering_mode_curr = 'MB'
                vl, va = self.MB_driver.steer()
            else:
                self.steering_mode_curr = 'CX'
                vl, va = self.steer2goal()
        if self.steering_mode_prev != 'reflex' and self.steering_mode_curr == 'reflex':
            self.MB_driver.learn_repulsive()
        elif self.steering_mode_prev == 'reflex' and self.steering_mode_curr != 'reflex':
            self.MB_driver.learn_attractive()
        if self.steering_mode_curr == 'CX':
            self.MB_driver.learn_route()
        return vl, va

    def _steer2point(self, goal):
        pos = self.odometry_sensor.sense_position()
        orn = self.odometry_sensor.sense_orientation()
        vec2goal = goal - pos
        orn2goal = np.arctan2(vec2goal[1], vec2goal[0])
        orn_diff = (orn2goal - orn) % (2 * np.pi)
        vl = np.cos(orn_diff)
        va = -np.sin(orn_diff)
        return vl, va

    def steer2goal(self):
        return self._steer2point(self.pos_goal)

    def steer2escape(self):
        return self._steer2point(self.obstacle_avoidance_goal)
