import numpy as np
import argparse as arg
#import matplotlib.pyplot as plt
str1 = input("Enter the location for the Training Data: ")
train_data = np.genfromtxt(str1,dtype = str,delimiter = ",", skip_header = 1)

ids = train_data[:,np.newaxis,0]

m = int(len(ids)*0.75)
ids_75 = ids[0:m,:]
ids_25 = ids[m:,:]

train_values = train_data[:,np.newaxis,1].astype(np.float32)

train_75 = train_values[0:m,:]
train_25 = train_values[m:,:]

train_ids = np.empty((m,0),int)

l1 = []
l2 = []

sum_x = 0.0
sum_xs = 0.0
sum_xy = 0.0
sum_y = 0.0

for i in range(0,m):
    date1 = ids_75[i][0].split("/")
    date1 = [int(numeric_string) for numeric_string in date1]
    l1.append(-date1[0]*date1[0] + date1[2]*date1[2])
    l2.append(-date1[0]*date1[0] + date1[2]*date1[2])
    sum_x += (-date1[0]*date1[0] + date1[2]*date1[2])
    sum_xs += (-date1[0]*date1[0] + date1[2]*date1[2])**2
    sum_xy += (-date1[0]*date1[0] + date1[2]*date1[2])*train_75[i][0]
    sum_y += train_75[i][0]
    
train_ids = np.append(train_ids,np.array([l1]).transpose(),axis = 1)


#Y = a + bX , Y = train_values ,X = tarin_ids
a = ((sum_y*sum_xs) - (sum_x*sum_xy))/((len(ids)*sum_xs) - (sum_x*sum_x))
b = ((len(ids)*sum_xy) - (sum_x*sum_y))/((len(ids)*sum_xs) - (sum_x*sum_x))

weights = []
weights.append(a)
weights.append(b)
weights = np.array(weights)

test_values_cv = np.empty((len(ids_25),0),float)
ls1_cv =[]
for i in range(0,len(ids_25)):
    date1 = ids_25[i][0].split("/")
    date1 = [int(numeric_string) for numeric_string in date1]
    ls1_cv.append(a + b * (-date1[0]*date1[0] + date1[2]*date1[2]))

test_values_cv = np.append(test_values_cv,np.array([ls1_cv]).transpose(),axis = 1)

mse = 0

for i in range(0,len(ids_25)):
    mse += (test_values_cv[i][0] - train_25[i][0])**2
mse/=len(ids_25)
print(mse)

lm = [(a+b*(l2[i])) for i in range(len(l2))]
#plt.scatter(l2,train_75.flatten(),label="Input points")
#plt.plot(l2,lm,label="Linear Model", color= "red")
#plt.xlabel('ids')
#plt.ylabel('values')
#plt.legend()
#plt.title("Linear Regression model")
#plt.show()

str2 = input("Enter loaction of Test data file: ")
test_data = np.genfromtxt(str2,dtype = str,delimiter = ",",skip_header = 0)

n = len(test_data)-1

test_dat = np.empty((len(test_data),0),str)
test_values = np.empty((n,0),float)
ls = ["id"]
ls1 = []

for i in range(1,n+1):
    ls.append(test_data[i])
    date1 = test_data[i].split("/")
    date1 = [int(numeric_string) for numeric_string in date1]
    ls1.append(a + b * (-date1[0]*date1[0] + date1[2]*date1[2]))

test_dat = np.append(test_dat,np.array([ls]).transpose(),axis = 1)
test_values = np.append(test_values,np.array([ls1]).transpose(),axis = 1)


final_test_value = np.array([["value"]])

test_str = test_values.astype(np.str_)

final_test_value = np.concatenate((final_test_value, test_str), axis = 0)

array = np.concatenate((test_dat,final_test_value),axis = 1 )

str3 = input("Enter the location for predicted_data file: ")
np.savetxt(str3, array, delimiter=',',comments='', fmt='%s')


print(f"weight={weights}")

#print(array)