import numpy as np

class Perceptron:
    def __init__(self,eta,epoch):
        self.weight=np.random.rand(3)*1e-4
        print(f"intital weight before training: \n{self.weight}")
        self.eta=eta
        self.epoch=epoch
            
    
    def activationfunction(self,input,weight):
        z=np.dot(input,weight)
        return np.where(z > 0,1,0)

    def fit(self,x,y):
        self.x=x
        self.y=y
        
        x_with_bias=np.c_[self.x,-np.ones((len(self.x),1))]
        print(f"x_with_bias \n{x_with_bias}")
        
        for epoch in range(self.epoch):
            print("_ _"*10)
            print(f"for epoch{epoch}")
            
            y_pre=self.activationfunction(x_with_bias,self.weight)
            print(f"prefect value after{epoch}{y_pre} ")
            self.error= self.y-y_pre
            print(f"error: \n {self.error}")
            self.weight=self.weight+self.eta*np.dot(x_with_bias.T,self.error)
            print("###"*10)
    
    def predict(self,x):
        x_with_bias=np.c_[x,-np.ones((len(x),1))]
        return self.activationfunction(x_with_bias,self.weight)
                
    
    
    def total_loss(self):
        total_loss=np.sum(self.error)
        return total_loss