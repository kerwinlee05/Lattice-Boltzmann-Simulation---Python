import numpy as np
from matplotlib import pyplot

plot_every = 25

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    Nx = 400  # lattice cell amount
    Ny = 100
    tau = 0.53  # the collision timescale
    Nt = 3500

    # lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1]) # x axis particles
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1]) # y axis particles
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]) # particles chances

    # initial condition
    F = np.ones([Ny, Nx, NL]) + .01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3 # value can be play around with it

    cylinder = np.full((Ny, Nx), False) # if false then empty space, if true then obstacle

    for y in range(0, Ny):
        for x in range(0, Nx):
            if(distance(Nx//4, Ny//2, x, y)<12): # distance of the cylinder (can modify)
                cylinder[y][x] = True

    # main loop
    for ut in range(Nt):
        print(ut)

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis= 0)

        boundaryF = F[cylinder, :]
        boundaryF = boundaryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]     # velocity to the touching opposite boundary

        # fluid variables

        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        F[cylinder] = boundaryF
        ux[cylinder] = 0                            # velocity in cylinder to 0
        uy[cylinder] = 0

        # collision

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux +cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
            )

        F = F + -(1/tau) * (F-Feq)

        if(ut%plot_every == 0):
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy
            pyplot.imshow(curl, cmap="bwr")
            pyplot.pause(.01)
            pyplot.cla()





if __name__ == "__main__":
    main()

