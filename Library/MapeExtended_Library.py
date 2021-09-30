class SVR_mapext:
    
    """
    SVR based on MAPE loss, and Elastic Net regularization. 
    Fits models where the values of the target variable are positive.
    
        -- Parameter --
            C: determines the number of points that contribute to creation of the boundary. 
               (Default = 0.1)
               The bigger the value of C, the lesser the points that the model will consider.
               
            epsilon: defines the maximum margin in the feature space (Default = 0.1).
                        The bigger its value, the more general~underfitted the model is.
                        
            lamda: controls the implication of the weighted Elastic Net regularization.
                        
            kernel: name of the kernel that the model will use. Written in a string format.
                    (Default = "linear"). 
        
                    acceptable parameters: 
                        "linear", "poly", "polynomial", "rbf", 
                        "laplacian", "cosine".
        
                    for more information about individual kernels, visit the 
                    sklearn pairwise metrics affinities and kernels user guide.
                    
                    https://scikit-learn.org/stable/modules/metrics.html
            
            Specific kernel parameters: 

        --Methods--
            fit(X, y): Learn from the data. Returns self.

            predict(X_test): Predicts new points. Returns X_test labels.

            coef_(): Returns alpha support vectors (sv) coefficient, X sv, and b.

            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Call the library
            >>> from MapeExtended_Library import SVR_mapext
            ...
            ## Initialize the SVR object with custom parameters
            >>> model = SVR_mapext(C = 10, kernel = "rbf", gamma = 0.1)
            ...
            ## Use the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predict with the given model
            >>> y_prediction = fitted_model.predict(X_test)
            ...
            ## e.g
            >>> print(y_prediction)
            np.array([12.8, 31.6, 16.2, 90.5, 28, 1, 49.7])
    
    """
    
    def __init__(self, C = 0.1, epsilon = 0.1, lamda = 0.2, kernel = "linear", **kernel_param):
        import numpy as np
        import cvxpy as cp
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils import check_X_y, check_array 
        self.cp = cp
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.pairwise_kernels = pairwise_kernels
        self.kernel_param = kernel_param
        self.check_X_y = check_X_y
        self.check_array = check_array
        self.lamda = lamda
        
    def fit(self, X, y):
        """ 
        Computes coefficients for new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contains labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points.
        """
        X, y = self.check_X_y(X, y)
        # hyperparameters
        C = self.C 
        epsilon =  self.epsilon
        lamda = self.lamda
        
        kernel = self.kernel
        pairwise_kernels = self.pairwise_kernels
        cp = self.cp
        # Useful parameters
        ydim = y.shape[0]
        onev = np.ones((ydim,1))
        
        # Matrices and constraints for the optimizer
        K = pairwise_kernels(X, X, metric = kernel, **self.kernel_param)
        A = onev.T
        b = 0.0
        G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
        h_ = np.concatenate((100*C*np.ones(ydim)/y, 100*C*np.ones(ydim)/y)); 
        h = h_.reshape(-1, 1)
        
        beta = cp.Variable((ydim, 1))
        Ev = (epsilon*y.T)/100
        
        # loss function and constraints
        min_fun = (1/2)*cp.quad_form(beta, K) - y.T @ beta + lamda*((1*y.T-Ev) @ cp.abs(beta) + Ev/2 @ beta**2)
        objective = cp.Minimize(min_fun)
        constraints = [A @ beta == b, G @ beta <= h]

        
        # Solver and solution
        prob = cp.Problem(objective,constraints)
        result = prob.solve()
        
        # Support vectors
        beta_1 = np.array(beta.value).reshape(-1)
        indx = abs(beta_1) > 5e-3
        beta_sv = beta_1[indx]
        x_sv = X[indx,:]
        y_sv = y[indx]
        
        # get w_phi and b
        k_sv = pairwise_kernels(x_sv, x_sv, metric = kernel, **self.kernel_param)
        cons = np.where(beta_sv >= 0, 1 - epsilon/100, 1 + epsilon/100)
        
        w_phi = beta_sv @ k_sv
        b = np.mean((y_sv*cons - w_phi)); self.b = b
        self.beta_sv = beta_sv; self.x_sv = x_sv
        return self
        
    def predict(self, X_):
        """Predicts new labels for a given set of new 
           independent variables (X_test).
           
           --Parameters--
               X_test: nxm matrix containing all the points that 
                       will be predicted by the model.
                       n is the number of points. m represents the
                       number of features/dimensions of each point.
            
           --Returns--
               a nx1 vector containing the predicted labels for the 
               input variables.
                
        """
        X_ = self.check_array(X_)
        k_test = self.pairwise_kernels(self.x_sv, X_, metric = self.kernel, **self.kernel_param)
        w_phi_test = self.beta_sv @ k_test
        predict = w_phi_test + self.b
        return predict
    
    def coef_(self):
        """--Returns--
                - dual support vectors
                - primal support vectors
                - intercept"""
        return self.beta_sv, self.x_sv, self.b