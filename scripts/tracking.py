import torch
import sound
import queue
import math

class Tracker:
    def __init__(self, finger1:int, finger2:int, kit:sound.Kit, sound:str, queue_max:int=3, buffer:float=0.1):
        self.kit = kit
        self.sound = sound
        self.qm = queue_max
        self.buff = buffer
        self.f1 = finger1
        self.f2 = finger2
        self.q = []
        self.qd = []
        self.bc = 0
        
        self.i = False

        self.touch = 0.1
    
    def update(self, hand, dt):
        if hand is None: return

        pi = self.i

        f1,f2 = hand[self.f1],hand[self.f2]
        d = ((f1[0]-f2[0])**2 + (f1[1]-f2[1])**2)**0.5
        
        self.q.insert(0,(d,dt))
        if len(self.q) > self.qm: self.q.pop()
        if len(self.q) < self.qm: return

        if self.bc < self.buff: self.bc += self.q[0][1]; return

        
        c1 = self.q[0][0] < 0.07
        c2 = self.q[-1][0] - self.q[1][0] > 0.05 or True
        c3 = self.q[0][0] < self.q[1][0]

        self.i = c1 and c2 and c3

        if self.i: self.kit.play(self.sound); self.bc = 0
            
