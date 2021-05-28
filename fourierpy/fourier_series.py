import numpy as np
from  pydantic import BaseModel, Field
from typing import List 

class FourierModel(BaseModel):
    a0: float = None
    a: List[float] = None 
    b: List[float] = None 
    n: int = Field(None, gt=0)
    l: float = Field(None, gt=0)
    
    class Config:
        validate_assignment = True
        extra = 'forbid'
        
    def fit(self,x,y,n):
        N = np.arange(1,n)
        L = (np.max(x) - np.min(x))
        print(L)
        dx = np.gradient(x)
        dx_r = dx.reshape(-1,1)

        n_r = N.reshape(-1,1)
        x_r = x.reshape(1,-1)/L
        y_r = y.reshape(1,-1)

        alpha = 2*np.pi * n_r *x_r

        a0 = (2/L) * np.dot(y,dx)
        a = (2/L)*np.dot(y*np.cos(alpha),dx_r)
        b = (2/L)*np.dot(y*np.sin(alpha),dx_r)
        
        self.a0 = a0
        self.a = np.squeeze(a).tolist()
        self.b = np.squeeze(b).tolist()
        self.n = n
        self.l = L

    def get_series(self,x):
        N = np.arange(1,self.n)
        a0 = self.a0
        L = self.l
        b = np.array(self.b)
        a = np.array(self.a)
        x = np.array(x)
        aa = np.dot(np.cos(2*np.pi*N*x.reshape(-1,1)/L),a.reshape(-1,1))
        bb = np.dot(np.sin(2*np.pi*N*x.reshape(-1,1)/L),b.reshape(-1,1))
        return a0*0.5 + aa + bb


