import numpy as np
import pickle
import statsmodels.api as sm
from PIL import Image

def purepython():
    list100 = list(range(100 + 1)) # Initiating the list

    sum_squared = (sum(list100))**2 # Taking the sum of all numbers and the square that sum
    squared_sum = sum([i**2 for i in list100]) # Taking the square of all numbers and then sum

    list_answer = abs(squared_sum - sum_squared) # Taking the difference
    return list_answer

def numpyarray():
    list100 = np.arange(100 + 1) # Initiating the numpy array

    sum_squared = np.sum(np.square(list100))
    squared_sum = np.square(np.sum(list100))

    list_answer = abs(squared_sum - sum_squared)
    return list_answer
    
def question1(num):

    # Function to create product of numbers 
    def list_product(digits, maximum):
        # Calculating the maximum of left, right, yp and down
        for i in range(digits.shape[0]):
            product = max(np.prod(digits[:,i]), # Horizontal
                          np.prod(digits[i,:]), # Vertical
                          np.prod(digits.diagonal()), # Diagonal
                          np.prod(np.fliplr(digits).diagonal()) # Reverse diagonal
                          )        
            # Checking whether its higher than the maximum
            if product > maximum:
                maximum = product

        return maximum

    # Length of the number of rows
    num_rows = 20
    adjacent_numbers = 4
    max_product = 0

    # List out of all numbers
    list_num = num.split()

    # Empty list 
    list_lists = []

    # Numpy list of lists - Changing the format from the a big list, to a list of lists
    for elem in range(num_rows):
        start = (elem) * 20 # The grid has 20 columns
        end = (elem + 1) * 20 
        row = list_num[start:end]
        list_lists.append(row)    

    # Using numpy for easier slicing
    list_lists = np.array(list_lists, dtype = int) # This is used for using the functions with less effort

    # Getting 4x4 matrices out of all the datapoints
    for i in range(num_rows - adjacent_numbers):
        # Going column wise 
        start_i = i
        end_i = i + 4
        # As well as row wise
        for j in range(num_rows - adjacent_numbers):
            start_j = j
            end_j = j + 4
            # Applying the function for every 4x4 matrix
            max_product = list_product(list_lists[start_i:end_i,start_j:end_j],
                                       max_product)

    print(max_product)
    
def beta_coefficients(): 
    # Loading the data
    with open('data/data.pkl','rb') as f:
        data = pickle.load(f)

    # Loading the Column names 
    column_names = ["Sale (in Dollars)", "Pack Size", "State Bottle Cost", "Packs Sold", "Bottle Volume (in ml)"]

    # Reshaping array from a 1x500000 format to a 5x100000 format
    reshaped_data = data.reshape(100_000,-1)

    # Changing the string varibles to floats
    float_data = reshaped_data.astype(np.float)

    # Separating the Sale variable from the rest
    independent = float_data[:,1:]
    Y = float_data[:,0]

    # Creating a column with only ones and add that to the numpy array as a column (this is done for the intercept)
    ones = np.ones(independent.shape[0])
    X = np.c_[ones, independent]

    # Applying regression coefficient formula
    X_prime = np.transpose(X)  

    inverse_part = np.linalg.inv(np.dot(X_prime, X))
    X_prime_Y = np.dot(X_prime, Y)
    beta = np.dot(inverse_part, X_prime_Y)

    # Printing the coefficients and the name of the regressor 
    return beta

def stats_package():
    # Loading the data
    with open('data/data.pkl','rb') as f:
        data = pickle.load(f)

    # Loading the Column names 
    column_names = ["Sale (in Dollars)", "Pack Size", "State Bottle Cost", "Packs Sold", "Bottle Volume (in ml)"]

    # Reshaping array from a 1x500000 format to a 5x100000 format
    reshaped_data = data.reshape(100_000,-1)

    # Changing the string varibles to floats
    float_data = reshaped_data.astype(np.float)

    # Separating the Sale variable from the rest
    independent = float_data[:,1:]
    Y = float_data[:,0]

    # Creating a column with only ones and add that to the numpy array as a column (this is done for the intercept)
    ones = np.ones(independent.shape[0])
    X = np.c_[ones, independent]
    
    # Defining statistical model
    model = sm.OLS(Y, X)
    
    # Fitting the results
    results = model.fit()
    
    # Printing the entire OLS summary statistics
    return results.summary()

def inverse_color(png_array):
    # Creating the inverse of the RGB color code
    inverse_array = 255 - png_array
    invimg = Image.fromarray(inverse_array)
    invimg.save('data/inverted.png')
    invimg.show()
    
def grayscale(png_array):
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    black_white = rgb2gray(png_array)
    invimg = Image.fromarray(black_white)
    new_p = invimg.convert("L")
    new_p.save('data/blackwhite.png')
    new_p.show()
    
def convolution(array):
    # Number of colors of the RGB color code 
    no_colors = array.shape[2]
    # Numbers of pixels of the picture
    no_pixels = array.shape[0]
    # Kernel chosen for this example
    kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                    ])
    # Flatten the kernel for the matrix multiplication
    flat_kernel = kernel.flatten()
    # Separating the colors
    red = array[:,:,0]
    green = array[:,:,1]
    blue = array[:,:,2]

    # Padding each matrix afterwards
    padded_red = np.pad(red, 1)
    padded_green = np.pad(green, 1)
    padded_blue = np.pad(blue, 1)
    
    # Slicing a matrix out of the big picture
    # Matrix size 
    sub_matrix_size = kernel.shape[0]

    # Looping ending 
    matrix_size = padded_red.shape[0]

    # Create empty container for taking on the values
    container = np.zeros([array.shape[0], array.shape[0]])

    for i in range(matrix_size - 2):
        for j in range(matrix_size - 2):
            matrix = padded_red[i:i + 3, j:j + 3].flatten()
            product = np.dot(flat_kernel, matrix)
            container[i,j] = product
    
