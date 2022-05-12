import numpy as np
from  pydantic import BaseModel, Field
from typing import List 

class FourierTrig(BaseModel):
    a0: float = Field(..., description="a0")
    a: List[float] = Field(..., description="a")
    b: List[float] = Field(..., description="b")
    n: int = Field(..., description="n", gt=0)
    l: float = Field(..., description="l", gt=0)
    
    class Config:
        validate_assignment = True
        extra = 'forbid'
    
    def __str__(self):
        return f'Trigonometric Fourier Series with {self.n} terms'

    def __repr__(self):
        return f'Trigonometric Fourier Series with {self.n} terms'
    
    @classmethod
    def fit(cls,x,y,n):
        N = np.arange(1,n)
        L = (np.max(x) - np.min(x))

        dx = np.gradient(x)
        dx_r = dx.reshape(-1,1)

        n_r = N.reshape(-1,1)
        x_r = x.reshape(1,-1)/L
        y_r = y.reshape(1,-1)

        alpha = 2*np.pi * n_r *x_r
        
        a0 = (2/L) * np.dot(y,dx)
        a = (2/L)*np.dot(y*np.cos(alpha),dx_r)
        b = (2/L)*np.dot(y*np.sin(alpha),dx_r)

        return cls(
            a0=a0,
            a=np.squeeze(a.astype(float)).tolist(),
            b=np.squeeze(b.astype(float)).tolist(),
            n=n,
            l=L
        )

    def forward(self,x):
        N = np.arange(1,self.n)
        a0 = self.a0
        L = self.l
        b = np.array(self.b)
        a = np.array(self.a)
        x = np.array(x)
        aa = np.dot(np.cos(2*np.pi*N*x.reshape(-1,1)/L),a.reshape(-1,1))
        bb = np.dot(np.sin(2*np.pi*N*x.reshape(-1,1)/L),b.reshape(-1,1))
        return a0*0.5 + aa + bb

class FourierExp(BaseModel):
    c0: float = Field(...)
    c: List = Field(...)
    n: int = Field(...)
    l: float = Field(...)

    class Config:
        validate_assignment = True
        extra = 'forbid'

    def __str__(self):
        return f'Exponential Fourier Series with {self.n} terms'

    def __repr__(self):
        return f'Exponential Fourier Series with {self.n} terms'
    
    @classmethod
    def fit(cls,x,y,n):
        N = np.arange(1,n)
        L = (np.max(x) - np.min(x))

        N = np.arange(1,n)
        L = (np.max(x) - np.min(x))

        dx = np.gradient(x)
        dx_r = dx.reshape(-1,1)

        n_r = N.reshape(-1,1)
        x_r = x.reshape(1,-1)/L
        y_r = y.reshape(1,-1)

        alpha = -2*np.pi * n_r *x_r

        c0 = (2/L) * np.dot(y,dx)
        c = (2/L)*np.dot(np.exp(1j*alpha)*y_r,dx_r)
        
        return cls(
            c0 = c0,
            c = np.squeeze(c).tolist(),
            n = n,
            l = L
        )
        
    def forward(self,x):
        N = np.arange(1,self.n)
        c0 = self.c0
        L = self.l
        c = np.array(self.c)
        x = np.array(x)
        
        alpha = 1j*2*np.pi * N*x.reshape(-1,1)/L

        
        return np.matmul(np.exp(alpha),c.reshape(-1,1)) + c0*0.5
        
        
        
        