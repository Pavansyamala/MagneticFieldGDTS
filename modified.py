import math
import numpy as np
import matplotlib.pyplot as plt
from MagneticField import MagneticField

class GDTS:

    def __init__(self, turn_rate, airspeed):
        self.turn_radius = airspeed / turn_rate
        self.turn_rate = turn_rate
        self.airspeed = airspeed
        self.strength = 10  # Strength at the Transmitters Location
        self.trans_loc = [25, 30 ,2]  # Transmitter Location
        self.tran_ori = [45 , 45 , 45]
        self.total_path_coordinates_x = []
        self.total_path_coordinates_y = []
        self.Initialization()

    # def signalstrength(self, pos):
    #     x, y = pos[0], pos[1]
    #     expont = -(((x - self.trans_loc[0])**2 / 200) + ((y - self.trans_loc[1])**2 / 200))
    #     curr_strength = self.strength * (math.exp(expont))
    #     return curr_strength 
    
    def magneticField(self,rec_loc , tra_loc , rec_ori , tra_ori ):
        
        obj = MagneticField(rec_loc , rec_ori , tra_loc , tra_ori) 
        return np.linalg.norm(obj.magnetic_field)

    def Initialization(self):
        self.strengths = []
        self.loop = 1
        self.timestep = 0
        self.del_t = 1/6
        self.freq = 6
        self.threshold_dist = 6
        self.uav_pos = []
        x, y = map(float, input("Enter initial X, Y coordinates separated by space of Receiver: ").split())
        self.initial_rec_loc = [x, y]
        self.curr_pos = [x, y]
        self.grad_dir = []
        self.x_vel = 0
        self.y_vel = 0
        self.p = [0, 0]
        self.w = self.turn_rate
        self.uav_pos.append(self.curr_pos)

        self.heading = 0 
        self.rec_ori = [0 , 0 , self.heading]
        self.delta = 2

        self.IterLoop()

    def GradientDirection(self, uav_pos, strengths):
        uav_pos = np.array(uav_pos)
        A = np.hstack((uav_pos, np.ones((len(strengths), 1))))
        B = np.array(strengths).reshape(-1, 1)
        gradient = (np.linalg.inv(A.T @ A) @ A.T) @ B
        print(f"Gradient: {gradient[:2].flatten()}")
        return gradient[:2].flatten()

    def IterLoop(self):
        curr_dist = (self.trans_loc[0] - self.uav_pos[-1][0])**2 + (self.trans_loc[1] - self.uav_pos[-1][1])**2

        x, y = self.curr_pos

        self.strengths.append(self.magneticField([x,y,2] , self.trans_loc , self.rec_ori , self.tran_ori))

        self.total_path_coordinates_x.append(x)
        self.total_path_coordinates_y.append(y)

        count = 1 

        while curr_dist > self.threshold_dist:
            self.timestep += 1

            if count > 30000 :
                break 

            count += 1 

            # Update heading
            self.heading = self.heading + (self.turn_rate * self.del_t)
            if self.heading > 180:
                self.heading -= 360
            elif self.heading < -180:
                self.heading += 360

            self.x_vel = self.airspeed * np.cos(np.radians(self.heading))
            self.y_vel = self.airspeed * np.sin(np.radians(self.heading))

            x += (self.x_vel * self.del_t)
            y += (self.y_vel * self.del_t)

            self.total_path_coordinates_x.append(x)
            self.total_path_coordinates_y.append(y)
            
            self.rec_ori[2] = self.heading 
            self.strengths.append(self.magneticField([x, y , 2] , self.trans_loc , self.rec_ori , self.tran_ori))
            self.uav_pos.append([x, y])

            self.p = np.array([x, y]) - np.array(self.curr_pos)

            print(f"Timestep: {self.timestep}, Heading: {self.heading}, Turn Rate: {self.turn_rate}, Current Position: ({x}, {y})")

            if self.loop == 1:
                if self.timestep > 3 and (self.strengths[self.timestep - 3] < self.strengths[self.timestep - 2] and self.strengths[self.timestep - 2] == self.strengths[self.timestep - 1] and self.strengths[self.timestep - 1] > self.strengths[self.timestep]):
                    self.turn_rate = -self.turn_rate
                    self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    self.timestep = 0
                    self.loop += 1
                    print("Turn rate changed due to condition 1")
                    continue 

                if self.timestep > 2 and (self.strengths[self.timestep - 2] < self.strengths[self.timestep - 1] and self.strengths[self.timestep - 1] > self.strengths[self.timestep]):
                    self.turn_rate = - self.turn_rate
                    self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    self.timestep = 0
                    self.loop += 1
                    print("Turn rate changed due to condition 2")
                    # break 

            else:

                if abs(np.degrees(np.arccos(np.dot(self.p, self.grad_dir) / (np.linalg.norm(self.p) * np.linalg.norm(self.grad_dir))))) < self.delta:
                    self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)

                    theta = np.degrees(np.arctan2(self.grad_dir[1], self.grad_dir[0]))

                    if (self.heading - theta) * self.turn_rate >= 0:
                        self.turn_rate = -self.turn_rate 
                    if np.linalg.norm(np.array(self.uav_pos[-1]) - np.array(self.curr_pos)) <= 0.1:
                        self.turn_rate = -self.turn_rate
                        print("Turn rate Changed due to Complete Circular Trajectory")


                    self.loop += 1
                    self.timestep = 0
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    print("Turn rate changed due to gradient direction")

            curr_dist = (self.trans_loc[0] - x)**2 + (self.trans_loc[1] - y)**2

        self.plotting_path()

    def plotting_path(self):
        x = np.linspace(self.trans_loc[0] , self.initial_rec_loc[0] , 1000)
        y = np.linspace(self.trans_loc[1] , self.initial_rec_loc[1] , 1000)
        plt.figure(figsize=(10, 10))
        plt.scatter(self.total_path_coordinates_x, self.total_path_coordinates_y, c='r', s=2)
        plt.scatter([self.trans_loc[0]], [self.trans_loc[1]], c='b', s=10)
        plt.scatter([self.initial_rec_loc[0]], [self.initial_rec_loc[1]], c='b', s=10)
        plt.plot(x , y , 'g')
        plt.annotate("Rec" , xy = (self.initial_rec_loc[0]-2 , self.initial_rec_loc[1]-2))
        plt.annotate("Tran" , xy = (self.trans_loc[0] , self.trans_loc[1]))
        plt.xlabel('X - coordinates ')
        plt.ylabel('Y - coordinates ')
        plt.title('Total Path Travelled Using GDTS ')
        plt.savefig("GDTS_test.png")
        plt.show()

if __name__ == '__main__':
    turn_rate = float(input('Enter the Turn Rate of Receiver (deg/sec): '))
    air_speed = float(input('Enter the Air Speed(m/s): '))
    alg = GDTS(turn_rate=turn_rate, airspeed=air_speed)