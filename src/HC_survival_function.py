"""
This script computes the survival function of the HC statistic for a given sample size n.
The survival function is computed using a simulation of the null distribution of the HC statistic.
We use the simulation results to fit a bivariate function of the form Pr[HC >= x | n] = f(n, x).
The simulation results are saved in a file named HC_null_sim_results.csv.
use function get_HC_survival_function to load the bivariate function or simulate the distribution. 
"""

import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline

import logging
import hashlib
import os
import sys
sys.path.append('../')
from src.fit_survival_function import fit_survival_func

NN = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500]  # values of n to simulate

def get_HC_survival_function(gamma, stbl,
                             HC_null_sim_file_prefix="HC_null_sim_results",
                              log_space=True, nMonte=10000,
                               nn=NN):

    # find hash of parameters:
    
    m = hashlib.md5()
    m.update((str(nMonte) + str(stbl) + str(gamma) + str(NN)).encode('utf-8'))
    hash = m.hexdigest()[:10]
    HC_null_sim_file = HC_null_sim_file_prefix + '_' + hash + ".csv"
    
    xx = {}
    # check if file exists:
    if os.path.isfile(HC_null_sim_file):
        logging.info(f"Loading HC null values from {HC_null_sim_file}...")
        df = pd.read_csv(HC_null_sim_file, index_col=0)
        for n in df.index:
            xx[n] = df.loc[n]
    else:
        logging.info("Simulated HC null values file was not found.")
        for n in tqdm(NN):
            logging.info(f"Simulating HC null values for n={n}...")
            yy = np.zeros(nMonte)
            for j in range(nMonte):
                uu = np.random.rand(n)
                mt = MultiTest(uu, stbl=stbl)
                yy[j] = mt.hc(gamma=gamma)[0]
            xx[n] = yy
        # save to file:
        df = pd.DataFrame(xx)
        df.index.name = "n"
        df.to_csv(HC_null_sim_file)
        logging.info(f"HC null values were saved to {HC_null_sim_file}.")

    xx0 = np.linspace(-1, 8, 57)
    zz = []
    for n in nn:
        univariate_survival_func = fit_survival_func(xx[n], log_space=log_space)
        zz.append(univariate_survival_func(xx0))
        
    func_log = RectBivariateSpline(np.array(nn), xx0, np.vstack(zz))

    if log_space:
        def func(x, y):
            return np.exp(-func_log(x,y))
        return func
    else:
        return func_log
    

def main():
    func = get_HC_survival_function(HC_null_sim_file_prefix=HC_NULL_SIM_FILE, stbl=STBL)
    print("Pr[HC >= 3 |n=50] = ", func(50, 3)[0][0]) # 9.680113e-05
    print("Pr[HC >= 3 |n=100] = ", func(100, 3)[0][0]) # 0.0002335
    print("Pr[HC >= 3 |n=200] = ", func(200, 3)[0][0]) # 0.00103771
    

if __name__ == '__main__':
    main()