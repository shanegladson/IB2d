
#-------------------------------------------------------------------------------------------------------------------#
#
# IB2d is an Immersed Boundary Code (IB) for solving fully coupled non-linear 
#   fluid-structure interaction models. This version of the code is based off of
#   Peskin's Immersed Boundary Method Paper in Acta Numerica, 2002.
#
# Author: Nicholas A. Battista
# Email:  nickabattista@gmail.com
# Date Created: May 27th, 2015
# Institution: UNC-CH
#
# This code is capable of creating Lagrangian Structures using:
#   1. Springs
#   2. Beams (*torsional springs)
#   3. Target Points
#   4. Muscle-Model (combined Force-Length-Velocity model, "HIll+(Length-Tension)")
#
# One is able to update those Lagrangian Structure parameters, e.g., spring constants, resting ##   lengths, etc
# 
# There are a number of built in Examples, mostly used for teaching purposes. 
# 
# If you would like us #to add a specific muscle model, please let Nick (nickabattista@gmail.com) know.
#
#--------------------------------------------------------------------------------------------------------------------#

import numpy as np

################################################################################################################
#
# def: Computes the components of the force term in Navier-Stokes from
#           arbitrary external forces, i.e., external force to get desired
#           velocity profile on fluid grid
#
################################################################################################################


def please_Compute_External_Forcing(dt,current_time,x,y, grid_Info, uX, uY, first, inds):

    #
    # dt:           time-step 
    # current_time: Current time of simulation (in seconds)
    # x:            x-Eulerian pts
    # y:            y-Eulerian pts
    # grid_Info:    holds lots of geometric pieces about grid / simulations
    # uX:           x-Velocity on Eulerian Grid
    # uY:           y-Velocity on Eulerian Grid


    # Grid Info #
    Nx =    grid_Info[0] # # of Eulerian pts. in x-direction
    Ny =    grid_Info[1] # # of Eulerian pts. in y-direction
    Lx =    grid_Info[2] # Length of Eulerian grid in x-coordinate
    Ly =    grid_Info[3] # Length of Eulerian grid in y-coordinate
    dx =    grid_Info[4] # Spatial-size in x
    dy =    grid_Info[5] # Spatial-size in y
    supp =  grid_Info[6] # Delta-def support
    Nb =    grid_Info[7] # # of Lagrangian pts. 
    ds =    grid_Info[8] # Lagrangian spacing


    # Stiffness for Arbitrary External Force to Fluid Grid
    kStiff = 1e4

    # Width of Channel
    w = Ly

    # Max Velocity Desired
    uMax = 0.12

    if first == 1:

        # Compute Where You Want to Apply Force
        xMin = 0.01
        xMax = 0.02
        yMin = 0.00
        yMax = Ly

        inds = give_Me_Indices_To_Apply_Force(x,y,xMin,xMax,yMin,yMax)
        first = 0

    # Compute External Forces from Desired Target Velocity
    fx, fy = give_Me_Velocity_Target_External_Force_Density(current_time,dx,dy,x,y,Nx,Ny,Lx,Ly,uX,uY,kStiff,w,uMax,inds)

    # Compute Total External Forces
    Fx = fx
    Fy = fy

    return (Fx, Fy, first, inds)




######################################################################################
#
# def: computes indices for exerting forces in specified places on fluid grid 
#
######################################################################################

def give_Me_Indices_To_Apply_Force(x,y,xMin,xMax,yMin,yMax):

    j=0 
    noMinYet = 1
    while noMinYet:

        if ( x[j] >= xMin ):
            iX_min = j
            noMinYet = 0
        
        j=j+1
    

    j=x.size - 1
    noMaxYet = 1
    while noMaxYet:

        if ( x[j] <= xMax ):
            iX_max = j
            noMaxYet = 0
        
        j=j-1
    

    j=0 
    noMinYet = 1
    while noMinYet:

        if ( y[j] >= yMin ):
            iY_min = j
            noMinYet = 0
        
        j=j+1

    j=y.size - 1
    noMaxYet = 1
    while noMaxYet:

        if ( y[j] <= yMax ):
            iY_max = j
            noMaxYet = 0
        
        j=j-1
    

    iX_Vec = np.arange(iX_min,iX_max+1,1)
    iY_Vec = np.arange(iY_min,iY_max+1,1)

    n = 0
    inds = np.zeros((len(iX_Vec)*len(iY_Vec),2))

    for i in range(0,iX_Vec.size):
        for j in range(0,iY_Vec.size):
            inds[n,0] = iX_Vec[i]
            inds[n,1] = iY_Vec[j]
            n = n+1 

    return inds


######################################################################################
#
# def: computes the External Force Densities! 
#
######################################################################################

def give_Me_Velocity_Target_External_Force_Density(t,dx,dy,x,y,Nx,Ny,Lx,Ly,uX,uY,kStiff,w,Umax,inds):

    # t:  current time in simulation
    # Nx: # of nodes in x-direction on Eulerian grid
    # Ny: # of nodes in y-direction on Eulerian grid
    # uX: x-Velocity on Eulerian grid
    # uY: y-Velocity on Eulerian grid
    # kStiff: stiffness parameter
    # inds: indices on the fluid grid for where to apply the arbitrary external force


    fx = np.zeros((Ny,Nx))     # Initialize storage for x-force density from EXTERNAL FORCES
    fy = np.zeros((Ny,Nx))     # Initialize storage for y-force density from EXTERNAL FORCES

    for n in range(0,inds.shape[0]):
        i = int(inds[n,0])
        j = int(inds[n,1])

        uX_Tar,uY_Tar = please_Give_Target_Velocity(t,dx,dy,x,y,Lx,Ly,i,j,w,Umax)    

        fx[j,i] = fx[j,i] - kStiff*( uX[j,i] - uX_Tar )
        fy[j,i] = fy[j,i] - kStiff*( uY[j,i] - uY_Tar )

    fx_exts = fx
    fy_exts = fy

    return (fx_exts, fy_exts)

    # MIGHT NOT NEED THESE!
    #fx_exts = fx/ds^2
    #fy_exts = fy/ds^2



########################################################################################################
#
# def: computes the Target Velocity Profile (MODEL DEPENDENT)
#
########################################################################################################

def please_Give_Target_Velocity(t,dx,dy,xGrid,yGrid,Lx,Ly,i,j,w,Umax):


    # t:     current time in simulation
    # dx:    x-Grid spacing
    # dy:    y-Grid spacing
    # xGrid: vector of xPts in Eulerian grid
    # yGrid: vector of yPts in Eulerian grid
    # Lx:    x-Length of Eulerian Grid
    # Ly:    y-Length of Eulerian Grid
    # i:     ith component in x-Grid
    # j:     jth component in y-Grid
    # w:     width of Channel
    # Umax:  maximum velocity

    y = yGrid[j];  # y-Value considered

    uX_Tar = -Umax * (100 * np.tanh(2*t)) * ( (Ly/2+w/2) - ( y ) )*( (Ly/2-w/2) - ( y ) ); # Only external forces in x-direction
    uY_Tar = 0;                                                                            # No external forces in y-direction


    return (uX_Tar, uY_Tar)
