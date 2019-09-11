#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import optimize as so
import math


# Q2.

# In[2]:


"""  0  1  2  3  4
     5  6  7  8  9
    10 11 12 13 14 
    15 16 17 18 19
    20 21 22 23 24
"""
""" 0 1 2 3
    4 5 6 7
    8 9 10 11
    12 13 14 15
"""

direction = {"N":-5,"S":5,"W":-1,"E":1} #direction to move (in 1d space) for given action in Q2
direction_ = {"N":-4,"S":4,"W":-1,"E":1} #direction to move (in 1d space) for given action in Q6
corners = {"N":[0,1,2,3,4],"S":[20,21,22,23,24],"W":[0,5,10,15,20],"E":[4,9,14,19,24]} #set of corners for each direction action for Q2
corners_ = {"N":[0,1,2,3],"S":[12,13,14,15],"W":[0,4,8,12],"E":[3,7,11,15]}  #set of corners for each direction action for Q2

#probability of choosing action a given we're in state s
def pi(a,s):
    return 1/4

#checks if the move is a special move
def is_special_move(s_,s,a):
    if(s==1 and s_==21):
        return True
    if(s==3 and s_==13):
        return True
    return False

#checks if the move is a corner move
def is_corner_move(s_,s,a):
    if(s==1 or s==3):
        return False
    if(s_==s and s in corners[a]):
        return True
    else:
        return False

#checks if reaching s_ from s by action is is valid or not (only for right and left corners)
def check(s_,s,a):
    if(s%5==0 and a=="W" and (s_+1)%5==0 and s==s_+1):
        return False
    if((s+1)%5==0 and a=="E" and (s_%5==0) and s+1==s_):
        return False
    return True

#checks if we can reach s_ from s by action a
def reachable(s_,s,a):
    if(is_special_move(s_,s,a)):
        return True
    if(is_corner_move(s_,s,a)):
        return True
    if(s!=1 and s!=3 and s_==s+direction[a] and check(s_,s,a)):
        return True
    return False

#returns the reward when we reach state s_ from s through action a
def reward(s,a,s_):
    if(is_special_move(s_,s,a)):
        return 10*(s-3)/(-2)+5*(s-1)/2
    if(is_corner_move(s_,s,a)):
        return -1
    return 0

#returns probability of reaching s_ with reward r from state s and action a
def p(s_,r,s,a):
    if(not reachable(s_,s,a)):
        return 0
    else:
        if(r==reward(s,a,s_)):
            return 1
        else:
            return 0

#main function to solve grid
#used linalg.solve function to solve the system of linear questions formed for v(s)
def grid_solve():
    gamma = 0.9
    # Solving AX=b, so below A=coeff and b=b. X is a vector which has value function for each state
    coeff = np.zeros((25,25)) #matrix for the 25 eqns of 25 states 
    b = np.zeros((25,1)) #the constant b of above eqn
    for i in range(coeff.shape[0]):
        s=i
        for a in "NSWE":
            prod = pi(a,s) #prob of picking the action which is 1/4
            for s_ in range(25):
                for r in [0,-1,10,5]:
                    prob = p(s_,r,s,a)*prod
                    b[i] += prob*r #adding this to the contant term of eqn i ie in b[i]
                    coeff[s,s_] -= gamma*prob #coeff of state s_ in eqn i(=s). Subtracting as initial values are zero and I want these on the other side as I have to make AX=b
        coeff[s,s] += 1 #Adding one to the coeff of state s for eqn i(=s) and we've this term on right side of equality
    ans = np.linalg.solve(coeff,b)
    print("The rounded values (like fig3.2) are")
    for i in range(5):
        for j in range(5):
            print(round(ans[i*5+j][0],1),end="\t")
        print()
        
        


# In[3]:


grid_solve()


# Q4. I tried but I wasn't able to solve the system of non linear equations

# Q6.

# In[14]:


#checks if reaching s_ from s by action is is valid or not (only for right and left corners)
def check_(s_,s,a):
    if(s%4==0 and a=="W" and (s_+1)%4==0 and s==s_+1):
        return False
    if((s+1)%4==0 and a=="E" and (s_%4==0) and s+1==s_):
        return False
    return True

#checks if we can reach s_ from s by action a
def reachable_(s_,s,a):
    if(s==0 or s==15):
        return 0
    if(s_==s and s in corners_[a]):
        return True
    if(s_==s+direction_[a] and check_(s_,s,a)):
        return True
    else:
        return False

#returns the reward when we reach state s_ from s through action a
def reward(s,a,s_):
    if(s_==0 or s_==15):
        return 0
    else:
        return -1

#returns probability of reaching s_ with reward r from state s and action a
def p_(s_,r,s,a):
    if(s==0 and s_==0 and r==0):
        return 1
    if(s==15 and s_==15 and r==0):
        return 1
    if(not reachable_(s_,s,a)):
        return 0
    else:
        if(r==reward(s,a,s_)):
            return 1
        else:
            return 0

#main function for q6
def grid():
    gamma = 0.9
    v_ = np.random.normal(size=(16,1))
    for i in range(0,4):
        for j in range(0,4):
            print(round(v_[4*i+j,0],2),end="\t")
        print()
    v_[0,0]=0
    v_[15,0]=0
    q_ = np.random.normal(size=(16,1))
    pi_n = np.random.randint(1,5,(16,1))
    pi_ = []
    for a in range(0,16):
        if(pi_n[a,0]==1):
            pi_.append("N")
        elif(pi_n[a,0]==2):
            pi_.append("S")
        elif(pi_n[a,0]==3):
            pi_.append("E")
        elif(pi_n[a,0]==4):
            pi_.append("W")
    delta = 0
    theta = 1e-5
    policy_stable = False
    while(not policy_stable):
        #Policy Evaluation
        while(delta<theta):
            delta = 0
            for state in range(0,16):
                v = v_[state,0]
                summ = 0
                for s_ in range(0,16):
                    for r_ in [-1,0]:
                        summ+=p_(s_,r_,state,pi_[state])*(r_+gamma*v_[s_,0])
                v_[state,0]=summ
                delta = max(delta,abs(v-v_[state,0]))
        for i in range(0,4):
            for j in range(0,4):
                print(round(v_[4*i+j,0],2),end="\t")
            print()
        #Policy Improvement
        policy_stable = True
        for state in range(0,16):
            old_action = pi_[state]
            argmax_a = ""
            argmax_a_val = math.inf*(-1)
            for action in "NSEW":
                ans = 0
                for s_ in range(0,16):
                    for r_ in [-1,0]:
                        ans+=p(s_,r_,state,action)*(r_+gamma*v_[s_,0])
                if(ans>argmax_a_val):
                    argmax_a_val = ans
                    argmax_a = a
            pi_[state]=argmax_a
            if(old_action!=pi_[state]):
                policy_stable=False
    for i in range(0,4):
        for j in range(0,4):
            print(round(v_[4*i+j,0],2),end="\t")
        print()
                    
    


# In[15]:


grid()


# In[ ]:




