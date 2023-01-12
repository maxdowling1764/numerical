from simulations.Lorenz import Lorenz
import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_series = []
    y_series = []
    z_series = []

    l1, = ax.plot3D(x_series, y_series, z_series, 'blue')
    
    initial_state = np.random.rand(1,3)*20
    sim = Lorenz(initial_state, 10.0, 28.0, 8.0/3.0,delta_t=0.01)
    print(sim.start())

    while(sim.running):
        sim.update(sim.dt)
        #print(sim.past_states)
        x_series.append(sim.curr_state[0][0])
        #print(sim.curr_state)
        y_series.append(sim.curr_state[0][1])

        z_series.append(sim.curr_state[0][2])
        l1.set_data_3d(x_series, y_series, z_series)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

if __name__ == '__main__':
    main()