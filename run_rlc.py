from simulations.NodalRLC import NodalRLC
import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_series = []
    y_series = []
    z_series = []

    R = np.array([[0.0, 100.0], [50.0, 0.0]])
    L = np.array([[0.0, 5.0], [2.0, 0.0]])
    C = np.array([[0.0, 0.5], [10.0, 0.0]])
    w_s = np.array([60.0, 0.0])
    V0 = np.array([120.0, 0.0])

    V_s = lambda t: V0*np.sin(w_s*t)

    A = np.array([0, 1],[1, 0])

    

    l1, = ax.plot3D(x_series, y_series, z_series, 'blue')
    
    initial_state = np.zeros(2,2,2)
    sim = NodalRLC(initial_state, )
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