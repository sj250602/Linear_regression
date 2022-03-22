import numpy as np
import argparse 


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
    
    Y = train_data[:,np.newaxis,1].astype(np.float32)
    
    list1 = []
    for i in range(0,len(x_data)):
        list1.append(x_data[i][0])
        
    list1 = np.array(list1)
    
    X = np.empty((len(x_data),0),float)
    for i in range (0,args.polynomial+1):
        list3 = np.power(list1,i)
        np.set_printoptions(suppress=True)
        X = np.append(X,np.array([list3]).transpose(),axis = 1)
    
    if(args.method=="pinv"):
    
        X_transpose = X.transpose()
        A = np.dot(X_transpose,X)
        A_inv = np.linalg.inv(A)
        B = np.dot(A_inv,X_transpose)
        B = np.dot(B,Y)
        y_predicted = np.dot(X,B)
        E = Y - y_predicted
        weights = B.flatten()
        print(f"weights={weights}")
        
    else:
    
        B = np.zeros((args.polynomial+1,1))
        E = np.zeros((len(X),1))
        iteration = 100000
        learning_rate = 0.0001
        for i in range(0,iteration):
            batch_list = create_batches(X,Y,args.batch_size)
            for batch in batch_list:
                X_mini,Y_mini = batch
                y_predicted = np.dot(X_mini,B)
                B = B - ((learning_rate*2)/args.batch_size)*np.dot(X_mini.transpose(),(y_predicted-Y_mini))
                E = Y_mini - y_predicted
                
                
        weights = B.flatten()
        print(f"weights={weights}")
        
if __name__ == '__main__':
    args = setup()
    demo(args)
