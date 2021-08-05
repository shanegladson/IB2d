#!/usr/bin/env python3

'''-------------------------------------------------------------------------

 IB2d is an Immersed Boundary Code (IB) for solving fully coupled non-linear 
 	fluid-structure interaction models. This version of the code is based off of
	Peskin's Immersed Boundary Method Paper in Acta Numerica, 2002.

 Author: Nicholas A. Battista
 Email:  nick.battista@unc.edu
 Date Created: May 27th, 2015\
 Python 3.5 port by: Christopher Strickland
 Institution: UNC-CH

 This code is capable of creating Lagrangian Structures using:
 	1. Springs
 	2. Beams (*torsional springs)
 	3. Target Points
	4. Muscle-Model (combined Force-Length-Velocity model, "HIll+(Length-Tension)")

 One is able to update those Lagrangian Structure parameters, e.g., 
 spring constants, resting lengths, etc
 
 There are a number of built in Examples, mostly used for teaching purposes. 
 
 If you would like us to add a specific muscle model, 
 please let Nick (nick.battista@unc.edu) know.

----------------------------------------------------------------------------'''

import numpy as np
import sys
# Path Reference to where Driving code is found #
sys.path.append('../../../IBM_Blackbox')
import IBM_Driver as Driver
from please_Initialize_Simulation import please_Initialize_Simulation

###############################################################################
#
# FUNCTION: main2d is the function that gets called to run the code. It
#           itself reads in paramters from the input2d file, and passes
#           them to the IBM_Driver function to run the simulation
#
###############################################################################

def main2d():
    
    '''This is the "main" function, which ets called to run the 
    Immersed Boundary Simulation. It reads in all the parameters from 
    "input2d", and sends them off to the "IBM_Driver" function to actually 
    perform the simulation.'''
    
    # NEW FORMAT #
    # READ-IN INPUT2d PARAMETERS #
    Fluid_Params, Grid_Params, Time_Params, Lag_Struct_Params, Output_Params, Lag_Name_Params = please_Initialize_Simulation()    

    Driver.main(Fluid_Params,Grid_Params,Time_Params,Lag_Struct_Params,Output_Params,Lag_Name_Params)
    
if __name__ == "__main__":
    main2d()