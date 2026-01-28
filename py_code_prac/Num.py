import numpy as np

# create a numpy array
# arr = np.array([1, 2, 3, 4, 5])
# x = arr.copy()
# arr[0] = 42
# print(arr)
# print(x)

#create a  array fill zero
# zerosArray = np.zeros(3)
# print(zerosArray)

#create a  array fill onces
# onceArray = np.ones((2,3))
# print(onceArray)

#create a  array with a specific value
# fullArray = np.full((2,3),7) #shape , value
# print(fullArray)


#creating sequence of numbers
# seqArray = np.arange(1,11,1) #start stop step
# print(seqArray)

# #calculating identity matrix
# identityMatrix = np.eye(4) #size
# print(identityMatrix)


#array propeties and Operations
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print("Array Shape: ", arr.shape)
# print("Array Size: ", arr.size)
# print("Array Data Type: ", arr.dtype)
# print("Array Dimensions: ", arr.ndim)
# print("Array Item Size (in bytes): ", arr.itemsize)
# print("Array Total Bytes: ", arr.nbytes)
# print("Array Transpose: \n", arr.T)
# print("Array Reshaped (3x2): \n", arr.reshape(3, 2))
# print("Array Flattened: ", arr.flatten())

# #real life examples of numpy
# # example plants disese detction data
# plant_data = np.array([[5.1, 3.5, 1.4, 0.2],
#                        [4.9, 3.0, 1.4, 0.2],
#                        [6.2, 3.4, 5.4, 2.3],      
#                        [5.9, 3.0, 5.1, 1.8]])
# # Calculate mean and standard deviation for each feature
# mean = np.mean(plant_data, axis=0)    
# std_dev = np.std(plant_data, axis=0)
# print("Mean of each feature: ", mean)
# print("Standard Deviation of each feature: ", std_dev)
# # Normalize the data
# normalized_data = (plant_data - mean) / std_dev   
# print("Normalized Data: \n", normalized_data)


# ARR_2D =np.array([[1,2,3],[3,4,5],[5,6,7]])
# print("2D Array: \n", ARR_2D)
# print("shap of 2D Array: ",ARR_2D.shape)
# print("Size of 2D Array: ", ARR_2D.size)
# print("Data Type of 2D Array: ", ARR_2D.dtype)
# print("Dimensions of 2D Array: ", ARR_2D.ndim)

#astype conversion
# arr_float = ARR_2D.astype(np.float64)
# print("Array with float data type: \n", arr_float)
# arr_str = ARR_2D.astype(np.str_)
# print("Array with string data type: \n", arr_str)
# arr_bool = ARR_2D.astype(np.bool_)
# print("Array with boolean data type: \n", arr_bool)

#Numpy Maths Operations
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# arr2 = np.array([[7, 8, 9], [10, 11, 12]])
# print(arr1*5)
# print(arr2+10)
# print("Addition: \n", arr1 + arr2)
# print("Subtraction: \n", arr2 - arr1)
# print("Multiplication: \n", arr1 * arr2)
# print("Division: \n", arr2 / arr1)
# print("Matrix Multiplication: \n", np.dot(arr1, arr2.T))
# print("Element-wise Square: \n", np.square(arr1))
# print("Square Root: \n", np.sqrt(arr2))

#Aggregation Functions
# print("Sum of all elements in arr1: ", np.sum(arr1))
# print("Mean of all elements in arr1: ", np.mean(arr1))
# print("Maximum value in arr2: ", np.max(arr2))

# print("Minimum value in arr2: ", np.min(arr2))
# print("Standard Deviation of arr1: ", np.std(arr1))

#Indexing & Slicing Arrays
# ARR_2D =np.array([[1,2,3],[3,4,5],[5,6,7]])
# print("Original Array: \n", ARR_2D)
# print("Element at (1,2): ", ARR_2D[1, 2]) # it show element on row 1 and column 2
# print("First Row: ", ARR_2D[0, :]) #it show row on one index
# print("Second Column: ", ARR_2D[:, 1]) #it show column on one index 
# print("Sub-array (rows 0-1, cols 1-2): \n", ARR_2D[0:2, 1:3])    # it show sub array from row 0 to 1 and column 1 to 2
# print("Elements greater than 3: ", ARR_2D[ARR_2D > 3]) # it show element which is greater than 3
# #Modifying Arrays
# ARR_2D[0, 0] = 10
# print("Modified Array (after changing element at (0,0) to 10): \n", ARR_2D) #it change element at row 0 and column 0 to 10
# ARR_2D[:, 1] = 20
# print("Modified Array (after changing second column to 20): \n", ARR_2D) #it change all element of column 1 to 20

#fancy indexing
# rows = np.array([0, 2])
# cols = np.array([1, 2]) 
# print("Fancy Indexing (elements at (0,1) and (2,2)): ", ARR_2D[rows, cols]) # it show element at row 0 column 1 and row 2 column 2


#Reshaping & Manipulation
# reshaped_array = ARR_2D.reshape(1, 9)
# print("Reshaped Array (1x9): \n", reshaped_array)
# #flatteming array
# flattened_array = ARR_2D.flatten()
# reval_array = ARR_2D.ravel()
# print("Raveled Array: ", reval_array) #view of original array
# print("Flattened Array: ", flattened_array) #copy of original array

#stacking arrays
# arr_a = np.array([[1, 2], [3, 4]])
# arr_b = np.array([[5, 6], [7, 8]])
# vertical_stack = np.vstack((arr_a, arr_b))
# horizontal_stack = np.hstack((arr_a, arr_b))    
# print("Vertical Stack: \n", vertical_stack)
# print("Horizontal Stack: \n", horizontal_stack)ARR_2D =np.array([[1,2,3],[3,4,5],[5,6,7]])
# ARR_2D =np.array([[1,2,3],[3,4,5],[5,6,7]])
# print("2D Array: \n", ARR_2D)
# new_2d_array= np.insert(ARR_2D, 3, [8,9,10], axis=0)
# print("Array after inserting new row at index 1: \n", new_2d_array)
# appemd_array = np.append(new_2d_array, [[11,12,13]], axis=0)
# print("Array after appending new row: \n", appemd_array)


# concate_array = np.concatenate((ARR_2D, new_2d_array), axis=0 )
# print("Array after concatenation: \n", concate_array)
# delte_array = np.delete(concate_array,2, axis=0 )
# print("Array after deleting row at index 2: \n", delte_array)


#BROADCASTING
prices = np.array([100, 200, 300,400, 500, 600])
print("Original Array: \n", prices)  
scalar = 10
final_prices = prices -(prices* scalar / 100)
print("Final Prices: \n", final_prices)