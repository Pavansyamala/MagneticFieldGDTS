import numpy as np 
import matplotlib.pyplot as plt


class MagneticField:

    def __init__(self, pos_rec , ori_rec , pos_tra , ori_tra):
        self.m = np.array([1,0,0]) 
        self.pr = np.array(pos_rec)
        self.pt = np.array(pos_tra)

        roll_t = (np.pi/180)*ori_tra[0]  # Roll angle of transmitter 
        pitch_t = (np.pi/180)*ori_tra[1]  # Pitch angle of transmitter 
        yaw_t = (np.pi/180)*ori_tra[2]  # Yaw angle of transmitter 

        R_t_to_i = np.array([
        [np.cos(pitch_t)*np.cos(yaw_t), np.sin(roll_t)*np.sin(pitch_t)*np.cos(yaw_t) - np.cos(roll_t)*np.sin(yaw_t), np.cos(roll_t)*np.sin(pitch_t)*np.cos(yaw_t) + np.sin(roll_t)*np.sin(yaw_t)],
        [np.cos(pitch_t)*np.sin(yaw_t), np.sin(roll_t)*np.sin(pitch_t)*np.sin(yaw_t) + np.cos(roll_t)*np.cos(yaw_t), np.cos(roll_t)*np.sin(pitch_t)*np.sin(yaw_t) - np.sin(roll_t)*np.cos(yaw_t)],
        [-np.sin(pitch_t), np.sin(roll_t)*np.cos(pitch_t), np.cos(roll_t)*np.cos(pitch_t)]
        ])

        roll_r = (np.pi/180)*ori_rec[0]  # Roll angle of Reciever 
        pitch_r = (np.pi/180)*ori_rec[1]  # Pitch angle of Reciever
        yaw_r = (np.pi/180)*ori_rec[2]   # Yaw angle of Reciever    

        R_r_to_i = np.array([
            [np.cos(pitch_r)*np.cos(yaw_r), np.sin(roll_r)*np.sin(pitch_r)*np.cos(yaw_r) - np.cos(roll_r)*np.sin(yaw_r), np.cos(roll_r)*np.sin(pitch_r)*np.cos(yaw_r) + np.sin(roll_r)*np.sin(yaw_r)],
            [np.cos(pitch_r)*np.sin(yaw_r), np.sin(roll_r)*np.sin(pitch_r)*np.sin(yaw_r) + np.cos(roll_r)*np.cos(yaw_r), np.cos(roll_r)*np.sin(pitch_r)*np.sin(yaw_r) - np.sin(roll_r)*np.cos(yaw_r)],
            [-np.sin(pitch_r), np.sin(roll_r)*np.cos(pitch_r), np.cos(roll_r)*np.cos(pitch_r)]
        ])

        self.magnetic_field = self.ARVA_Field(self.pr , self.pt , R_t_to_i , R_r_to_i , self.m)  

    def ARVA_Field(self , pr , pt , R_t_to_i, R_r_to_i, m_vec):
        R_i_to_t = np.transpose(R_t_to_i)
        r = pr - pt
        r = np.dot(R_i_to_t, r)
        A = np.array([2*r[0]**2 - r[1]**2 - r[2]**2 , 3*r[0]*r[1] ,3*r[0]*r[2] ]).reshape(-1,1) 
        Am = np.dot(R_t_to_i, A)
        Am_x = Am[0, 0] 
        Am_y = Am[1, 0] 
        Am_z = Am[2, 0] 
        m_mag = np.linalg.norm(m_vec)
        rd = np.linalg.norm(r)
        H = np.array([(m_mag/(4*np.pi*rd**5))*Am_x, (1/(4*np.pi*rd**5))*Am_y, (1/(4*np.pi*rd**5))*Am_z])
        R_i_to_r = np.transpose(R_r_to_i)
        Hb = np.dot(R_i_to_r, H)
        return Hb  
    
    

if __name__ == '__main__':

    pt = np.array([5,5,2])
    ori_r = [45 , 45 , 45]
    ori_t = [45 , 45 , 45]
    pr = []
    field = []

    X = np.linspace(0,10,100)
    Y = np.linspace(0,10,100)

    field = []
    pr = []

    count = 1

    for x in X: 
        for y in Y: 
            obj = MagneticField([x,y,2] , ori_r , pt , ori_t) 
            pr.append([x,y,2]) 
            field.append(obj.magnetic_field)
            print(count) 
            count += 1   

    pr = np.array(pr)
    field = np.array(field) 

    plt.figure(figsize=(8, 8))
    plt.quiver(pr[:,0], pr[:,1],field[:,0], field[:,1], color='b', scale=20)
    plt.title('Magnetic Field (Bx, By)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

