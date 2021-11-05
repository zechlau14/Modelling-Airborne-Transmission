# Modelling-Airborne-Transmission
Implements the airborne transmission model in Lau et al. 2021 (see https://arxiv.org/abs/2012.12267). The code here will calculate the concentration of infectious particles,  infection risk and Time to Probable Infection (TTPI) in a room with recirculating flow.


The code requires the installation of the following python libraries:

(a) Numpy

(b) Scipy

(c) Matplotlib


To use the code: 

(1) Download the repository. 

(2) Edit the required .py file to change the parameters to your desired values. 

(3) Run the .py file.


Specific files:

C-contour.py: produces a contour plot for the concentration in a room at a chosen time.

C-point.py: For a chosen point in the room, produces a concentration versus time graph.

Prob-contour.py: produces a contour plot for the probability of infection in a room at a chosen time.

Prob-point.py: For a chosen point in the room, produces a probability versus time graph.

avg-Prob.py: Plots the spatially-average probability of infection in a room versus time.

TTPI-contour.py: produces a contour plot for the Time to Probable Infection (TTPI) in a room.

TTPI-vs-lambda.py: For a chosen point in the room, produces a TTPI versus Air exchange rate graph. 

TTPI-vs-R.py: For a chosen point in the room, produces a log-log plot of TTPI versus R (the particle emission rate). 


Contact information: mrzech.com@gmail.com
