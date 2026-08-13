import math
class Value:
    
    def __init__(self,data,_children=(),_op=''):
         self.data = data
         self._prev = set(_children)
         self._op = _op
         self.grad = 0
         self._backward = lambda : None
    
    
    def __repr__(self):
         return f"Value({self.data})"
    
     
    def __add__(self, other):
        if not isinstance(other,Value):
            other = Value(other)
            
        out = Value(self.data + other.data,
                     (self,other),
                     '+')
         
        def _backward():
             self.grad += out.grad
             other.grad += out.grad 
        
        out._backward = _backward
         
        return out
    
     
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        out = Value(self.data * other.data,
                    (self, other),
                    '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
            
        return out
    
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other,
                    (self,),
                    '**')
        
        def _backward():
            self.grad += (other*((self.data)**(other-1))) * out.grad
            
        out._backward = _backward
        return out
        
        
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        if not isinstance(other,Value):
            other = Value(other)               
        return self + (-other)
    
    def __rmul__(self, other):            
        return self * other
    
    def __truediv__(self, other):
        if not isinstance(other,Value):
            other = Value(other)
        return self*(other**-1)
    
    def __radd__(self, other):
        return self + other
    

    def tanh(self):
        out = Value(math.tanh(self.data),
                    (self,),
                    'tanh')
        
        def _backward():
            self.grad += (1 - (out.data)**2) * out.grad
            
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(child):
            if child not in visited:
                visited.add(child)
                for node in child._prev:       
                    build_topo(node)
            
                topo.append(child)
        
        build_topo(self)
    
        self.grad = 1
        for o in reversed(topo):
            o._backward()
        