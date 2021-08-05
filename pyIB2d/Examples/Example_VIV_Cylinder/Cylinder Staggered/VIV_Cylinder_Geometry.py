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

from math import cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

def CylinderGeometry():

    #
    # Grid Parameters (MAKE SURE MATCHES IN input2d !!!)
    #
    Nx = 200        # # of Eulerian Grid Pts. in x-Direction (MUST BE EVEN!!!)
    Ny = 200        # # of Eulerian Grid Pts. in y-Direction (MUST BE EVEN!!!)
    Lx = 0.1        # Length of Eulerian Grid in x-Direction
    Ly = 0.1        # Length of Eulerian Grid in y-Direction


    # Immersed Structure Geometric / Dynamic Parameters #
    ds = min(Lx/(2*Nx), Ly/(2*Ny))  # Lagrangian Spacing
    L = Lx # Length of channel
    w = Ly # Width of channel

    x0 = Lx/5.0 # x-center for cylinder
    y0 = Ly/4.0 # y-center for cylinder

    x1 = Lx/5.0
    y1 = 2*Ly/4.0

    x2 = Lx/5.0 # x-center for cylinder
    y2 = 3 * Ly/4.0 # y-center for cylinder

    x3 = 2*Lx/5.0
    y3 = 3*Ly/8.0

    x4 = 2 * Lx / 5.0  # x-center for cylinder
    y4 =  5*Ly/8.0  # y-center for cylinder

    x5 = 3 * Lx / 5.0
    y5 = 1 * Ly / 4.0

    x6 = 3 * Lx / 5.0  # x-center for cylinder
    y6 = 2 * Ly / 4.0  # y-center for cylinder

    x7 = 3 * Lx / 5.0
    y7 = 3 * Ly / 4.0

    x8 = 4 * Lx / 5.0  # x-center for cylinder
    y8 = 3*Ly/8.0  # y-center for cylinder

    x9 = 4 * Lx / 5.0
    y9 = 5*Ly/8.0

    r = 0.005 # radii for cylinder
    struct_name = 'viv_geo' # Name for .vertex, .spring, etc files.


    # Call function to construct geometry
    xLag,yLag = give_Me_Immsersed_Boundary_Geometry(ds,L,w,Lx,Ly)
    x0Lag_Cy, y0Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x0, y0)
    x1Lag_Cy, y1Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x1, y1)
    x2Lag_Cy, y2Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x2, y2)
    x3Lag_Cy, y3Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x3, y3)
    x4Lag_Cy, y4Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x4, y4)
    x5Lag_Cy, y5Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x5, y5)
    x6Lag_Cy, y6Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x6, y6)
    x7Lag_Cy, y7Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x7, y7)
    x8Lag_Cy, y8Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x8, y8)
    x9Lag_Cy, y9Lag_Cy = give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x9, y9)

    xLag_Cy = np.concatenate((x0Lag_Cy, x1Lag_Cy, x2Lag_Cy, x3Lag_Cy, x4Lag_Cy, x5Lag_Cy, x6Lag_Cy, x7Lag_Cy, x8Lag_Cy, x9Lag_Cy))
    yLag_Cy = np.concatenate((y0Lag_Cy, y1Lag_Cy, y2Lag_Cy, y3Lag_Cy, y4Lag_Cy, y5Lag_Cy, y6Lag_Cy, y7Lag_Cy, y8Lag_Cy, y9Lag_Cy))

    xTether = x0
    indsTether, x0_new = give_Me_Tethering_Pt_Indices(xLag, xTether)
    xLag_Cy = xLag_Cy + (x0_new - x0)

    # Plot Geometry to test
    # plt.plot(xLag[:len(xLag)//2],yLag[:len(xLag)//2],'r-')
    # plt.plot(xLag[len(xLag)//2:],yLag[len(xLag)//2:],'r-')
    plt.plot(xLag_Cy,yLag_Cy,'r-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, Lx, 0, Ly])
    plt.show(block=True)


    # Prints .vertex file!
    print_Lagrangian_Vertices(xLag_Cy, yLag_Cy,struct_name)

    # Prints .spring file!
    k_Spring_Tether = 1e4
    k_Spring = 2.0e7
    resting_length_tether = 2*r
    offset = 0
    print_Lagrangian_Springs(xLag_Cy, yLag_Cy, k_Spring, ds, r, offset, indsTether, resting_length_tether, k_Spring_Tether, struct_name)

    # Prints .beam file!
    k_Beam = 5.0e9
    C = compute_Curvatures(xLag_Cy, yLag_Cy)
    print_Lagrangian_Beams(xLag_Cy, yLag_Cy, k_Beam, C, struct_name, offset)


    # Prints .target file!
    k_Target = 2.5e7
    print_Lagrangian_Target_Pts(xLag_Cy,k_Target,struct_name)

########################################################################
#
# FUNCTION: prints VERTEX points to a file called "struct_name".vertex
#
########################################################################

def print_Lagrangian_Vertices(xLag,yLag,struct_name):

    N = len(xLag)
    
    with open(struct_name + '.vertex','w') as vertex_fid:
        vertex_fid.write('{0}\n'.format(N))

        #Loops over all Lagrangian Pts.
        for s in range(N):
            X_v = xLag[s]
            Y_v = yLag[s]
            vertex_fid.write('{0:1.16e} {1:1.16e}\n'.format(xLag[s],yLag[s]))

    
########################################################################
#
# FUNCTION: prints TARGET points to a file called "struct_name".target
#
########################################################################

def print_Lagrangian_Target_Pts(xLag,k_Target,struct_name):

    N = len(xLag)    # Total Number of Lagrangian Pts
    
    with open(struct_name + '.target','w') as target_fid:
        target_fid.write('{0}\n'.format(N))

        #Left Bottom Target Points
        for s in range(N):
            target_fid.write('{0} {1:1.16e}\n'.format(s,k_Target))
    
    
#####################################################################################
#
# FUNCTION: prints BEAM (Torsional Spring) points to a 
#        file called "struct_name".beam
#
#####################################################################################

def print_Lagrangian_Beams(xLag, yLag, k_Beam, C, struct_name, offset):

    # k_Beam: beam stiffness
    # C: beam curvature
    
    N = len(xLag) - 2 # NOTE: Total number of beams = Number of Total Lag Pts. - 2
    
    with open(struct_name + '.beam','w') as beam_fid:
        beam_fid.write('{0}\n'.format(N))

        #BEAMS BETWEEN VERTICES
        for s in range(N):
            if ( (s>0) and (s <= N-2) ):
                beam_fid.write('{0} {1} {2} {3:1.16e} {4:1.16e}\n'.format(
                    s-1+offset,s+offset,s+1+offset,k_Beam,C[s]))
            elif (s == 0):
                beam_fid.write('{0} {1} {2} {3:1.16e} {4:1.16e}\n'.format(
                    N+offset,s+offset,s+1+offset,k_Beam,C[s]))
            elif (s == N-1):
                beam_fid.write('{0} {1} {2} {3:1.16e} {4:1.16e}\n'.format(
                    N-1+offset, N+offset, 1+offset, k_Beam, C[s]))

########################################################################
#
# FUNCTION: prints SPRING points to a file called rubberband.spring
#
########################################################################

def print_Lagrangian_Springs(xLag,yLag,k_Spring,ds_Rest,r,offset,indsTether,resting_length_tether,k_Spring_Tether,struct_name):

    N = len(xLag) #Number of Lagrangian Pts. Total
    
    with open(struct_name + '.spring','w') as spring_fid:
        spring_fid.write('{0}\n'.format(N+N/2+2))

        #SPRINGS BETWEEN VERTICES
        for s in range(N):
            if s < N-2:
                ds_Rest = sqrt( ( xLag[s] - xLag[s+1] )**2 +  ( yLag[s] - yLag[s+1] )**2 )
                spring_fid.write('{0} {1} {2:1.16e} {3:1.16e}\n'.format(
                    s+offset,s+1+offset, k_Spring, ds_Rest))
            else:
                ds_Rest = sqrt( ( xLag[s] - xLag[1] )**2 +  ( yLag[s] - yLag[1] )**2 )
                spring_fid.write('{0} {1} {2:1.16e} {3:1.16e}\n'.format(
                    s+offset,1+offset,k_Spring,ds_Rest))
    
        for s in range(N//2):
            ds_Rest = sqrt( ( xLag[s] - xLag[s+N//2] )**2 +  ( yLag[s] - yLag[s+N//2] )**2 )
            spring_fid.write('{0} {1} {2:1.16e} {3:1.16e}\n'.format(
                    s+offset,s+N/2+offset,k_Spring,ds_Rest))

########################################################################
#
# FUNCTION: creates the Lagrangian structure geometry
#
########################################################################

def give_Me_Immsersed_Boundary_Geometry(ds,L,w,Lx,Ly):
    
    x = np.arange((Lx-L)/2,(L+(Lx-L)/2),ds)
    yBot = (Ly-w)/2*np.ones(x.size)
    yTop = (Ly - (Ly-w)/2)*np.ones(x.size)

    xLag = np.concatenate((x, x))
    yLag = np.concatenate((yBot, yTop))

    return (xLag,yLag)

########################################################################
#
# FUNCTION: creates the Lagrangian structure geometry for cylinder
#
########################################################################

def give_Me_Cylinder_Immersed_Boundary_Geometry(ds, r, x0, y0):
    dtheta = ds/(2*r)
    theta = -np.pi/2
    i = 0
    xLag = np.zeros(int(np.ceil(2*np.pi/dtheta)))
    yLag = np.zeros(int(np.ceil(2*np.pi/dtheta)))

    while theta < 3*np.pi/2:
        xLag[i] = x0 - r*cos(theta)
        yLag[i] = y0 - r*sin(theta)
        theta += dtheta
        i += 1
    return xLag, yLag

def give_Me_Tethering_Pt_Indices(xLag, xTether):
    indx = np.zeros(2, dtype=int)
    i = 0
    go = 1
    while go == 1:
        if xLag[i] > xTether:
            indx[0] = i
            go = 0
        i += 1

    nexti = len(xLag)//2
    indx[1] = indx[0]+nexti
    x0_new = xLag[indx[0]]
    return indx, x0_new


def compute_Curvatures(xLag, yLag):
    N = len(xLag)
    C = np.zeros(N)

    for i in range(N):
        if ((i > 0) and (i < N-1)):
            Xp = xLag[i-1]
            Xq = xLag[i]
            Xr = xLag[i+1]

            Yp = yLag[i-1]
            Yq = yLag[i]
            Yr = yLag[i+1]

        elif i == 0:
            Xp = xLag[N-1]
            Xq = xLag[i]
            Xr = xLag[i+1]

            Yp = yLag[N-1]
            Yq = yLag[i]
            Yr = yLag[i+1]

        elif i==N-1:
            Xp = xLag[N-2]
            Xq = xLag[N-1]
            Xr = xLag[0]

            Yp = yLag[N-2]
            Yq = yLag[N-1]
            Yr = yLag[0]

        C[i] = (Xr-Xq)*(Yq-Yp) - (Yr-Yq)*(Xq-Xp)
    return C
    
if __name__ == "__main__":
    CylinderGeometry()