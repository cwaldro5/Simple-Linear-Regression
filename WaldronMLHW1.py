# 1)
import numpy as np
import matplotlib.pyplot as plt

# 2)
xData = np.array([12., 72., 32., 47., 29., 37., 57., 64., 9., 22.])
yData = (2 * xData) + 5 + (4 * np.random.random())
#xData = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
#yData = 2*xData+50+5*np.random.random()

# 3)
bias = np.arange(0,100,1) #bias
weight = np.arange(-5, 5,0.1) #weight
Z = np.zeros((len(bias),len(weight)))
 
for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        Z[j][i] = 0        
        for n in range(len(xData)):
            Z[j][i] = Z[j][i] + (w*xData[n]+b - yData[n])**2 
        Z[j][i] = Z[j][i]/len(xData)

plt.xlim(0,100)
plt.ylim(-5,5)
plt.contourf(bias,weight,Z, 50, alpha =0.5,
cmap = plt.get_cmap('jet'))


# 4)
#wArray = np.zeros((len(x_data)))
  
b = 0
w = 0
lr = .00001 # was .00015
iteration = 100000 # was 10000

b_history = [b]
w_history = [w]

for i in range(iteration):
   bGrad = 0.0
   wGrad = 0.0
   for n in range(len(xData)):      
      bGrad = bGrad + (b + w * xData[n] - yData[n])* 1.0
      wGrad = wGrad + (b + w * xData[n] - yData[n]) * xData[n]
   
   w = w - lr * wGrad
   b = b - lr * bGrad
   b_history.append(b)
   w_history.append(w)
   
   
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5,color='black')

plt.show()
      
endLoss = 0
for i in range(len(xData)):
   endLoss = endLoss +(w * xData[n] + b -yData[n])**2
print("When b = ")
print(b)
print("And w = ")
print(w)
print("Total loss is :")
print(endLoss)  
