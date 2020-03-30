from scipy.optimize import curve_fit
import matplotlib.pyplot as mp
import pandas as pd
import numpy as np
import math

def test(x,a):
	return a*np.exp(-1*a*x)

n = 120000

X  = pd.read_csv('dataset.csv')
x  = X['Time']

x1 = np.array(x)
x1 = x1[1:]

x2 = np.insert(x1,0,0)
x2 = x2[:-1]

x3 = x1-x2		#Inter arrival time
x3 = x3[:n]		#First 1,20,000
x3.sort()		#Sort on the basis of inter-arrival time length

inter_count=[0]

#Helps to define intervals
check_n = 10**(-6)
check_i = 4

#First time_interval slot
time_inter = [check_i*check_n]
	
for i in range(n):
	if(x3[i]<check_i*check_n):
		inter_count[-1]+=1							#Increment packet to this inter_arrival time slot
	else:
		while(x3[i]>=check_i*check_n):
			check_i+=1
		time_inter.append(check_i*check_n)			#Make new inter_arrvial slot
		inter_count.append(1)						#intialize it with 1

inter_count = np.array(inter_count)
time_inter = np.array(time_inter)

#Whole data set inter arrival times
mp.plot(time_inter,inter_count)
mp.title('Inter-arrival times - Whole dataset')
mp.xlabel('Time intervals')
mp.ylabel('No. of packets')
mp.savefig('wholedata')
mp.show()

#Small subset of values to find lambda
mp.plot(time_inter[8:500],inter_count[8:500])
mp.title('Inter-arrival times - Sampled Dataset')
mp.xlabel('Time intervals')
mp.ylabel('No. of packets')
mp.savefig('sampleData')
mp.show()

lmbda = curve_fit(test, time_inter[8:500], inter_count[8:500])

print('Possible Value of Lambda : ',lmbda[0])
Y_axis = np.exp(-1*lmbda[0]*time_inter)

#Values with estimated lambda
mp.plot(time_inter,Y_axis)
mp.title('Fitted Curve using estimated lambda')
mp.xlabel('Time intervals')
mp.ylabel('PDF')
mp.savefig('fit')
mp.show()