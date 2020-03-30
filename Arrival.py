import matplotlib.pyplot as mp
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import math
from scipy.special import factorial
from collections import Counter


X  = pd.read_csv('dataset.csv')
x  = X['Time']

x3 = np.array(x)
n = len(x3)

tim = 0						#Time counter
incr_val = 10**(-2)

pack_count=[0]
time_slot = [tim + incr_val]

for i in range(n):
	if(x3[i]< tim):
		pack_count[-1]+=1
	else:
		while(x3[i]>= tim):
			tim += incr_val
		time_slot.append(tim)
		pack_count.append(1)

# print(len(time_slot))

mp.plot(time_slot,pack_count)
mp.title('No. of packets per time interval')
mp.xlabel('Time intervals')
mp.ylabel('Number of packets')
mp.savefig('poisson_1')
mp.show()

dic = Counter(pack_count)
probs = []*len(dic)
# print(dic)

for i in range(1,len(dic)+1):
	probs.append(dic[i])

probs = np.array(probs)
x_axis = np.array(range(1,len(dic)+1))


mp.plot(x_axis,probs/sum(probs))
mp.title('Poisson Arrivals for t = 10^-2')
mp.xlabel('No. of packets')
mp.ylabel('Probability of Arrival')
mp.savefig('poisson_2')
mp.show()



