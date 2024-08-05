import numpy as np 

import matplotlib.pyplot as plt


class TransmitterFrameField:

    def __init__(self, r , r0):
        m = np.array([1,0,0])
        # Calculate R and norm_R
        R = np.subtract(np.transpose(r), r0).T
        norm_R = np.sqrt(np.einsum("i...,i...", R, R)) 

        # Handle the case where the norm_R is zero to avoid division by zero
        norm_R[norm_R == 0] = np.inf

        # Calculate A (the magnetic field vector)
        m_dot_r = np.einsum("i,i...->...", m, R)
        self.B = (3 * R * m_dot_r / norm_R**5 - m[:, np.newaxis, np.newaxis, np.newaxis] / norm_R**3)  

if __name__ == "__main__":
    x = np.linspace(-50,50,100)
    y = np.linspace(-50 ,50,100) 
    z = np.zeros(100) 
    X , Y , Z = np.meshgrid(x, y, z) 
    trans_loc = [5,10,0]
    obj = TransmitterFrameField([X,Y,Z] , trans_loc) 
    Bx , By , _ = obj.B 

    plt.figure(figsize=(8,8))
    plt.streamplot(X[:,:,0] , Y[:,:,0] ,Bx[:,:,0] , By[:,:,0]) 

    plt.scatter([trans_loc[0]] , [trans_loc[1]] , c = 'm' , s = 100 ,label = 'Transmitter') 

    plt.xlabel('X-coordianets') 
    plt.ylabel('Y-coordianets') 

    plt.title("Magnetic Field Due to ARVA in Transmitter Mode") 
    plt.legend() 
    
    plt.savefig("TransmitterField.png")
    plt.show()