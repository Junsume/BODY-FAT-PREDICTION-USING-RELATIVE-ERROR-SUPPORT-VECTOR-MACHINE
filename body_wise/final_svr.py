import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("bodyfat.csv")

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['BodyFat', 'Density'])  # Features (all columns except 'BodyFat')
y = df['BodyFat']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[7]:


import numpy as np

class IRE_SVM:
    def __init__(self, C=4*10**7, gamma_b=1):
        self.C = C
        self.gamma_b = gamma_b
        self.alpha = None
        self.bias = None
        
    def gaussian_kernel(self, x1, x2, sigma2=4*10**6):
        # Assuming a Gaussian (RBF) kernel here
        return np.exp(-np.linalg.norm(x1 - x2)*2 / (2 * sigma2))
    
    def fit(self, X, y):
        N = len(y)
        K = self.gaussian_kernel(X, X)
        
        y = np.array(y).reshape(-1, 1)
        ones = np.ones((N, 1))
        I = np.eye(N)
        yT_y = np.matmul(y.T, y)
        
        # Compute A matrix
        A = np.block([
            [K + yT_y * self.C**-1 * I, ones],
            [ones.T,  -self.gamma_b]
        ])
        
        # Compute alpha and bias
        alpha_bias = np.linalg.solve(A, np.concatenate([y, [[0]]]))
        self.alpha = alpha_bias[:-1]
        self.bias = alpha_bias[-1]
        
    def predict(self, X):
        """
        Predict the target values for the given input data X.
        """
        N = len(self.alpha)
        y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            x = X[i]
            kernel_sum = 0
            for j in range(N):
                kernel_sum += self.alpha[j] * self.gaussian_kernel(X_train.values[j], x)
            y_pred[i] = kernel_sum + self.bias

        return y_pred
