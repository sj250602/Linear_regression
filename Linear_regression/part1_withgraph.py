import numpy as np
import argparse 
import matplotlib.pyplot as plt


def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=75, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=int, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()

def create_batches(X, Y, batch_size):

    batch_list = []
    merge = np.hstack((X,Y))
    np.random.shuffle(merge)
    i = 0
    for i in range(int(len(X)/batch_size)):
        X_mini = merge[i*batch_size:(i+1)*batch_size,:-1]
        Y_mini = merge[i*batch_size:(i+1)*batch_size,-1:]
        batch_list.append((X_mini, Y_mini))
    if len(X) % batch_size != 0:
        X_mini = merge[(i+1)*batch_size:len(X),:-1]
        Y_mini = merge[(i+1)*batch_size:len(X),-1:]
        batch_list.append((X_mini, Y_mini)) 
    return batch_list

def demo(args):
    train_data = np.genfromtxt(args.X,dtype = str,delimiter = ",")
    
    x_data = train_data[:,np.newaxis,0].astype(np.float32)
    
    m = int(len(x_data)*0.80)
    ids_75 = x_data[0:m,:]
    ids_25 = x_data[m:,:] 
    
    Y = train_data[:,np.newaxis,1].astype(np.float32)
    
    train_75 = Y[0:m,:]
    train_25 = Y[m:,:] 
    
    list1 = []
    for i in range(0,len(ids_75)):
        list1.append(ids_75[i][0])
        
    list1 = np.array(list1)
    
    X = np.empty((len(ids_75),0),float)
    for i in range (0,args.polynomial+1):
        list3 = np.power(list1,i)
        np.set_printoptions(suppress=True)
        X = np.append(X,np.array([list3]).transpose(),axis = 1)
        
    list12 = []
    for i in range(0,len(ids_25)):
        list12.append(ids_25[i][0])
        
    list12 = np.array(list12)
    
    X_test = np.empty((len(ids_25),0),float)
    for i in range (0,args.polynomial+1):
        list3 = np.power(list12,i)
        np.set_printoptions(suppress=True)
        X_test = np.append(X_test,np.array([list3]).transpose(),axis = 1)    
    
    if(args.method=="pinv"):
    
        X_transpose = X.transpose()
        A = np.dot(X_transpose,X)
        A_inv = np.linalg.inv(A)
        B = np.dot(A_inv,X_transpose)
        B = np.dot(B,train_75)
        y_predicted = np.dot(X,B)
        E = train_75 - y_predicted
        weights = B.flatten()
        print(f"weights = {weights}")
        
        y_nb = np.dot(X_test,B)
        mse = 0
        for i in range(len(ids_25)):
            mse+=(y_nb[i][0]-train_25[i][0])**2
        mse/=len(ids_25)
        print("{:.8f}".format(mse))
        
        mse1 = 0 
        for i in range(len(E)):
            mse1+=E[i][0]**2
        mse1/=len(E)
        print("{:.8f}".format(mse1))
        
        dictionary2 = {}
        for _ in range(len(ids_75)):
            dictionary2[ids_75[_][0]] = y_predicted[_][0]
        x = ids_75.flatten()
        x.sort()
        predict = []
        for _ in range(len(ids_75)):
            predict.append(dictionary2[x[_]])
        
        plt.scatter(ids_75.flatten(),train_75.flatten(),label="Input points")
        plt.plot(x,predict,label="Pinv", color= "red")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.title(f"Degree {args.polynomial}")
        plt.show()
        
        plt.plot(x,noise,label="Noise")
        plt.xlabel('x')
        plt.ylabel("Noise")
        plt.legend()
        plt.title(f"Degree {args.polynomial}")
        
        plt.show()
    else:
    
        B = np.zeros((args.polynomial+1,1))
        E = []
        iteration = 100000
        learning_rate = 0.0001
        for i in range(0,iteration):
            batch_list = create_batches(X,train_75,args.batch_size)
            for batch in batch_list:
                X_mini,Y_mini = batch
                y_predicted = np.dot(X_mini,B)
                B = B - ((learning_rate*2)/args.batch_size)*np.dot(X_mini.transpose(),(y_predicted-Y_mini))
                E.append(Y_mini - y_predicted)
                
                
                
        weights = B.flatten()
        print(f"weights = {weights}")
        
        y_nb = np.dot(X_test,B)
        mse = 0
        for i in range(len(ids_25)):
            mse+=(y_nb[i][0]-train_25[i][0])**2
        mse/=len(ids_25)
        print("{:.8f}".format(mse))
        
        
        dictionary1 = {}
        y_prd = np.dot(X,B)
        dictionary2 = {}
        for _ in range(len(ids_75)):
            dictionary1[ids_75[_][0]] = E[_][0]
            dictionary2[ids_75[_][0]] = y_prd[_][0]
        x = ids_75.flatten()
        x.sort()
        noise = []
        predict = []
        for _ in range(len(ids_75)):
            noise.append(dictionary1[x[_]])
            predict.append(dictionary2[x[_]])
        
        plt.scatter(ids_75.flatten(),train_75.flatten(),label="Input points")
        plt.plot(x,predict,label="Gd", color= "red")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.title(f"Degree {args.polynomial}")
        plt.show()
        
        plt.plot(x,noise,label="Noise")
        plt.xlabel('x')
        plt.ylabel("Noise")
        plt.legend()
        plt.title(f"Degree {args.polynomial}")
        
        plt.show()
        
        
if __name__ == '__main__':
    args = setup()
    demo(args)
