from LorenzSimulation import LorenzSimulation
from RLCSimulation import RLCSimulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def main():
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_series = []
    y_series = []
    z_series = []

    l1, = ax.plot3D(x_series, y_series, z_series, 'blue')

    #sim = LorenzSimulation(np.random.rand(1,3)*20, 10.0, 28.0, 8.0/3.0,delta_t=0.01)
    sim = RLCSimulation(np.zeros((2,1)), V0=50.0, w=71.17625e6, R=100.0, L=5.0, C=1.0, delta_t=0.00005)
    print(sim.start())

    while(sim.running):
        sim.update(sim.dt)
        #print(sim.past_states)
        x_series.append(sim.t)
        y_series.append(sim.curr_state[0][0])
        z_series.append(0.0)
        l1.set_data_3d(x_series, y_series, z_series)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

if __name__ == '__main__':
    main()