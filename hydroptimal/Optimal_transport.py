import numpy as np
import matplotlib.pyplot as plt

def optimal_transport(times_d,data,times_m,model,Sinkhorn_dist=False,k1=.2,k2=1.2,logeps=5,reg=1,NumIterMax=10000,cost=True):
    """
        
    Compute the optimal transport distance between two positive and possibly unbalanced times series data and model
    
    The first time series d is to be the data time series (that doesn't change through the inversion), because of a 
    preprocessing step that has to be the same during all the inversion.
 
 
    Parameters
    ---------- 
    times_d: List of the time steps of the data time series
    data: List of the data values along times_d
    times_m: List of the time steps of the model time series
    model: List of the model values along times_m  
    Sinkhorn_dist: If True, use the Sinkhorn algorithm, if False use the Wasserstein+L2 distance
    k: Float, Preprocessing "min value". Has to be strictly greater than 0. The higher k is, the faster the convergence is 
        reached, but the less sensitive to small variations the distance is. Default is 0.2
    eps: Float, Entropy regularisation factor for the Sinkhorn algorithm. The higher k is, the faster the convergence is reached,
        but the more biaised it is. Default is 1e-4
    reg: Float, Penalty factor given to high divergence between the Transport map marginals and the times series. The higher it
        is, the more the Sinkhorn algorithm will match the Wasserstein+L2 distance behaviour. The lower it is, the more it will
        match a Kullback-Leibler divergence behaviour. Default is 1
    StopTresh: Float, Stopping criteria for the Sinkhorn algorithm. Default is 1e-10
    NumIterMax: Number of Maximum Iteration of the Sinkhorn Algorithm. Default is 10000
    cost: bolean, if True return the cost, if False, return the Sinkhorn plan (only if Sinkhorn_dist=True)
    Returns
    -------
    float, optimal transport distance between the two time series using Sinkhorn distance or L1+Wasserstein distance
    """


    
    def preprocessing(times,a,max_time,max_value,k1,k2,how='minmax_scaling'):
        """ 
        Preprocess the time series so that the times are between 0 and 1 and the amplitude between k and 1+k for the data time
        series.
        
        Parameters
        ----------
        times: List or numpy Array of time steps
        a: List of times series value
        k: Float, Preprocessing mean value
        max_value: Float, Max value of the data 
        max_time: Float, Max value of data times
        
        Returns
        -------
        two List containting the times steps and the associated time series preprocessed according to the data
        
        """
        if how=='minmax_scaling':
            return times/max_time, k1+ a*(k2-k1)/max_value    
    
    def trapeze(x,y):
        """
        Compute the integral of a function using the trapeze method
        Parameters
        ----------
        x: List of xs
        y: List of ys
        
        Returns
        -------
        float, value of the integral
        """
        return sum((x[1:]-x[:-1])*(y[:-1]+y[1:])/2)
    
    def cumul(X,P):
        """
        Compute the cumulative function of a probabilistic distribution

        Parameters
        ----------

        X : List that contains the values list
        P : List that contains the Density probability asociated to the values list

        Returns
        -------
        res : List that contains the cumulative distribution
        """
        res=np.zeros(len(P))
        for i in range(len(res)):
            if P[i]<0:
                print('Valeur négative rencontrée')
            res[i]=trapeze(X[:i],P[:i])
        return res

    
    
    def poids(dist1,dist2):
        """
        Interpolation function used in inv_cumul
        """
        p1,p2=abs(dist1),abs(dist2)
        return (p1/(p1+p2)),(p2/(p1+p2))




    def WD(time_A,A,time_B,B,p=2,gamma=1,precision=1e-3):

        """
        Compute the Wasserstein+L1 distance 

        Parameters
        ----------

        times_A: List of the time steps of the data time series
        A: List of the data values along times_d
        times_B: List of the time steps of the model time series
        B: List of the model values along times_m  

        p : Wasserstein power 

        gamma : Weight given to the mass unbalances (Default 1, same as reg)

        precision : Precision on the inverse cumulative. Default: 0.001

        Returns
        -------
        shape_error+mass_error : Float, Wasserstein+L1 distance 

    

        """

        A_norm=A/trapeze(time_A,A)
        B_norm=B/trapeze(time_B,B)
        cumul_A=cumul(time_A,A_norm)
        cumul_B=cumul(time_B,B_norm)
        
        int_step=np.sort([*cumul_A,*cumul_B])
        
        inv_A=np.interp(int_step,cumul_A,time_A)
        inv_B=np.interp(int_step,cumul_B,time_B)
        
        
        shape_error=trapeze(int_step,(abs(inv_A-inv_B)**p))**(1/p)
        mass_error=gamma*(trapeze(time_A,A)-trapeze(time_B,B))**2
        return shape_error*trapeze(time_A,A)+mass_error
    
    


    def cost_matrix(ta,tb,how='L2'):
        """
        Compute the Euclidian cost matrix between two Times list
        
        Parameters
        ----------
        ta: List, contains the times steps for the first time series
        tb: List, contains the times steps for the second time series
        how: Method to compute the distance between two time steps
        
        Returns
        -------
        Numpy array, cost matrix
        """

        Ta,Tb=np.meshgrid(ta,tb)

        return (Ta-Tb)**2

    def Kullback_Leibler(a,b):
        """
        Compute the Kullback-Leibler divergence between two times series of same time steps
        The two times series shall not contain numbers <=0
        
        Parameters
        ----------
        a: List, first time series
        b: List, second time series
        
        Returns
        -------
        float, KL divergence between a and b
        
        """
        return sum(a*np.log(a/b)-a+b)
    
    def Sinkhorn(a,b,C,eps=1e-2,stopTrh=1e-10,NumIterMax=10000,cost=True): 
        """
        Compute the Sinkhorn algorithm for optimal transport

        Parameters
        ----------
        a: List, First distribution
        b: List, Second distribution
        C: Matrix, Cost matrix associated to the two time series
        eps: float, entropic regularisation parameter
        stopTrh: float, Treshold at which the progress is too small, and therefore where we have reached the convergence
        NumIterMax: int, In case of no convergence, maximum number of iteration
        cost: if True, return the cost, if False, return 

        """
        dim_a=len(a)
        dim_b=len(b)
        niter=int(-np.log(eps)/np.log(2))
        eps_i=1
        K=np.exp(-C/eps_i)
        v=np.ones(dim_b)
        if np.any(K@v == 0):
            print('probleme',np.max(C))
        u=a/(K@v)
        for j in range(niter):
            K=np.exp(-C/eps_i)

            P=u[:, None] * K * v[None, :]
            for i in range(NumIterMax):
                if (np.any(K.T@u == 0) or (np.any(K@v == 0))):
                    # we have reached the machine precision
                    # come back to previous solution and quit loop
                    break
                u=a/(K@v)
                v=b/(K.T@u)
                P2=u[:, None] * K * v[None, :]

                if np.sum(abs(P2-P))<stopTrh:
                    break
                P=P2

            eps_i/=2

        K=np.exp(-C/eps)
        P=u[:, None] * K * v[None, :]
        for i in range(NumIterMax):
            if (np.any(K.T@u == 0) or (np.any(K@v == 0))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                break

            u=a/(K@v)
            v=b/(K.T@u)
            P2=u[:, None] * K * v[None, :]

            if np.sum(abs(P2-P))<stopTrh:
                break
            P=P2

        if cost:
            return np.sum(P*C)
        else:
            return P

    
    def Sinkhorn_unb(a,b,C,logeps=8,reg=1,stopTrh=1e-8,NumIterMax=1000,cost=True):
        """
        Compute the Sinkhorn algorithm for unbalanced optimal transport
        
        Parameters
        ----------
        a: List, First time series
        b: List, Second times series
        C: Matrix, Cost matrix associated to the two time series
        eps: float, entropic regularisation parameter
        reg: float, unbalanced regularisation parameter
        stopTrh: float, Treshold at which the progress is too small, and therefore where we have reached the convergence
        NumIterMax: int, In case of no convergence, maximum number of iteration
        cost: if True, return the cost, if False, return 
        
        """

        dim_a=len(a)
        dim_b=len(b)
        eps=1
        K=np.exp(-C/eps)
        v=np.ones(dim_b)

        for j in range(2*logeps):
            eps/=np.sqrt(10)
            
            K=np.exp(-C/eps)

            for i in range(NumIterMax):
                u=a/(K@v)**(reg/(reg+eps))
                v=b/(K.T@u)**(reg/(reg+eps))
                
                
                
        P=u[:, None] * K * v[None, :]
        
        if cost:
            return np.sum(P*C)+reg*Kullback_Leibler(P@np.ones(dim_a),a)+reg*Kullback_Leibler(P.T@np.ones(dim_b),b)
        else:
            return P
        
    times_d_p,d_p=preprocessing(times_d,data,np.max(times_d),np.max(data),k1,k2)
    times_m_p,m_p=preprocessing(times_m,model,np.max(times_d),np.max(data),k1,k2)
    

    if Sinkhorn_dist:

        C=cost_matrix(times_d_p,times_m_p)

        Sink=Sinkhorn_unb(d_p,m_p,C,logeps=logeps,reg=reg,NumIterMax=NumIterMax,cost=cost)
        if cost:
            return Sink*np.max(data)
        else:
            return Sink
    else:
        
        return WD(times_d_p,d_p,times_m_p,m_p,gamma=reg)*np.max(data)
            
