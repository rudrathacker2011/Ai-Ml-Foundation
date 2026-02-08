#Here is all details of numpy library
import numpy as np 

#declaring 1-d array using numpy
arr_1d = np.array([1,2,3,4,5,6,7,8])
#declaring 2-d array using numpy
arr_2d = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
#about shape and dimension
print(arr_1d.shape)
print(arr_2d.shape)
print(arr_1d.ndim)
print(arr_2d.ndim)
print(arr_2d.sum(axis=0))#axis=0 means operation down the rows
print(arr_2d.sum(axis=1))#axis=1 means operation across the columns
print("\n")
#indexing & slicing
print(arr_1d[0])
print(arr_1d[-1])#imagine array like clock or fold array
print(arr_1d[2:7])#slice array form 3 to 7
print("\n")
print(arr_2d[0,1])
print(arr_2d[:,1])
print(arr_2d[1,:])
print("\n")
#vectorization(alternative of loop)
result = arr_1d*2;
print(result)
print("\n")
#reshaping
reshaped = arr_1d.reshape(2,4)
print(reshaped)
print(arr_1d.reshape(4,-1))#-1 use for auto-calculation
print("\n")
#broadcasting
add_this = np.array([1,2,3])
result_add = add_this + arr_2d;
print(result_add)