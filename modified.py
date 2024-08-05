import math
import numpy as np
import matplotlib.pyplot as plt
from TransmitterFrame import TransmitterFrameField
from MagneticField import MagneticField

class GDTS:

    def __init__(self, turn_rate):
        self.turn_radius = 10 # turn radius
        self.turn_rate = np.degrees(turn_rate)
        self.airspeed = self.turn_radius * turn_rate 
        self.trans_loc = np.array([150, 200 , 0])  # Transmitter Location
        self.tran_ori = [45 , 45 , 45]
        self.total_path_coordinates_x = []
        self.total_path_coordinates_y = []
        self.Initialization()
    
    def magneticField(self,rec_loc , tra_loc , rec_ori , tra_ori ):
        
        obj = MagneticField(rec_loc , rec_ori , tra_loc , tra_ori) 
        return np.linalg.norm(obj.magnetic_field)

    def Initialization(self):
        self.strengths = []
        self.loop = 1
        self.timestep = 0
        self.del_t = 1/self.turn_radius
        self.threshold_dist = self.turn_radius/2
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
        curr_dist = np.sqrt((self.trans_loc[0] - self.uav_pos[-1][0])**2 + (self.trans_loc[1] - self.uav_pos[-1][1])**2)

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
            self.heading = (self.heading + 180) % 360 - 180

            self.x_vel = self.airspeed * np.cos(np.radians(self.heading))
            self.y_vel = self.airspeed * np.sin(np.radians(self.heading))

            x += (self.x_vel * self.del_t)
            y += (self.y_vel * self.del_t)

            self.total_path_coordinates_x.append(x)
            self.total_path_coordinates_y.append(y)
            
            self.rec_ori[2] = self.heading 
            self.strengths.append(self.magneticField([x, y , 0] , self.trans_loc , self.rec_ori , self.tran_ori))
            self.uav_pos.append([x, y])

            self.p = np.array([x, y]) - np.array(self.curr_pos)

            print(f"Timestep: {self.timestep}, Heading: {self.heading}, Turn Rate: {self.turn_rate}, Current Position: ({x}, {y})")

            if self.loop == 1:

                if self.timestep > 2 and (self.strengths[self.timestep - 2] < self.strengths[self.timestep - 1] and self.strengths[self.timestep - 1] > self.strengths[self.timestep]):
                    self.turn_rate = - self.turn_rate
                    self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    self.timestep = 0
                    self.loop += 1
                    print("Turn rate changed due to condition 1")

            else:

                if abs(np.degrees(np.arccos(np.dot(self.p, self.grad_dir) / (np.linalg.norm(self.p) * np.linalg.norm(self.grad_dir))))) < self.delta:
                    self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)

                    theta = np.degrees(np.arctan2(self.grad_dir[1], self.grad_dir[0]))

                    if np.sign(self.heading - theta) == np.sign(self.turn_rate):
                       self.turn_rate = -self.turn_rate
                       print(f"Turn rate Changed After {self.loop} loop") 

                    if np.linalg.norm(np.array(self.uav_pos[-1]) - np.array(self.curr_pos)) <= 0.9:
                        self.turn_rate = -self.turn_rate
                        self.grad_dir = self.GradientDirection(self.uav_pos, self.strengths)
                        print("Turn rate Changed due to Complete Circular Trajectory")


                    self.loop += 1
                    self.timestep = 0
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    print("Turn rate changed due to gradient direction")

            curr_dist = np.sqrt((self.trans_loc[0] - x)**2 + (self.trans_loc[1] - y)**2)

        self.plotting_path()

    def plotting_path(self):


        a = np.linspace(self.trans_loc[0]-50,self.trans_loc[0]+50 , 100)
        b = np.linspace(self.trans_loc[1]-50,self.trans_loc[1]+50, 100)
        c = np.full(shape=(50,) , fill_value=self.trans_loc[2])
        A , B , C = np.meshgrid(a,b,c)

        obj = TransmitterFrameField([A,B,C],self.trans_loc)
        Bx , By , _ = obj.B 

        plt.figure(figsize=(10, 10))

        plt.streamplot(A[:,:,0], B[:,:,0], Bx[:,:,0] , By[:,:,0]) 

        plt.scatter(self.total_path_coordinates_x, self.total_path_coordinates_y, c='r', s=2) 

        plt.scatter([self.trans_loc[0]], [self.trans_loc[1]], c='m', s=100 , label='Transmitter') 
        plt.scatter([self.initial_rec_loc[0]], [self.initial_rec_loc[1]], c='g', s=100, label='Reciever') 


        plt.xlabel('X - coordinates ')
        plt.ylabel('Y - coordinates ')
        plt.legend()

        plt.title('Total Path Travelled Using GDTS ')
        plt.savefig("SimulationResults/GDTS_test5_3.png")
        plt.show()

if __name__ == '__main__':
    turn_rate = float(input('Enter the Turn Rate of Receiver (rad/sec): '))
    alg = GDTS(turn_rate=turn_rate)