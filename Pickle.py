# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:16:28 2021

@author: João Matos
"""

import pickle

#%% Definition of function that creates pickle file

def createPickleFile(variable, pickleName):
    PIK = pickleName + ".dat"

    with open(PIK, "wb") as f:
        pickle.dump(variable, f)
        

#%% Definition of function that gets pickle file

def getPickleFile(pickleName):
    PIK = pickleName + ".dat"

    with open(PIK, "rb") as f:
        return(pickle.load(f))