import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle
from jpda_agent import jpda_single
# from util import EKFcontrol, measurement, track
import random
import copy

def make_ellipse(mean, cov, gating_size):
    """Support function for scatter_ellipse."""
    

    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    if np.isclose(u[0,0], 0):
        angle = np.sign(u[0,1]) * 0.5 * np.pi
    else:
        angle = np.arctan(u[0, 1]/u[0,0])
    angle = 180 * angle / np.pi # convert to degrees
    v = 2 * np.sqrt(v * gating_size) #get size corresponding to level
    return Ellipse(mean[:2], v[0], v[1], 180 + angle, facecolor='none',
                  edgecolor='green',
                  #ls='dashed',  #for debugging
                  lw=1.5)

def generate_obs(t):
    z_k = []
    beta = np.pi/100
    # 1. ellipse 1
    x = 25 + 32.17*np.cos(t*beta) - 20.25*np.sin(t*beta)
    y = 25 + 32.17*np.cos(t*beta) + 20.25*np.sin(t*beta)
    target = [.4 * x, .4 *y]
    # plot_track(target)
    z_k.append(target)

    # 2. line
    x = 0.8*(t - 25)
    y = 0.8*(t - 25)
    z_k.append([x,y])


    # 3. line
    x = -.8 *(t - 25)
    y = .8 * (t - 25)
    z_k.append([x,y])
    # # 1. sin wave y = 25 + 3sinx, as GV2
    # z_k.append(x + 0.1 *np.random.normal(0, input_var)+ number1)
    # z_k.append(3 * np.sin(x) + 0.1 * np.random.normal(0, input_var) + 25 + number2)
    # # 2. parabola  y = .01 x^2, as sth else
    # z_k.append(x + 0.1 *np.random.normal(0, input_var)+ number1)
    # z_k.append(0.01 * x**2  + number2)
    
    # 3. some noises
    noise_k = []
    for i in range(random.randint(1, 5)):
        ran_point = [50* random.random() -25, 50* random.random()-25]
        noise_k.append(ran_point)
        

    z_k += noise_k
    return z_k, noise_k


def main():
    # single sensor multi target tracking applying jpda


    # 1. initialization of parameters and objective of jpda
    
    shape = ["rectangle", [50,50]]   # height & width of FoV of the rectangle
    sensor_para = {"position": [0, 0, 0],   # x, y, orientation of rectangle
            "shape": shape}

    dt = 0.1    # update rate 10 hz
    
    # in practical usage on hardware, we don't need to set isSimulation to true
    mtt = jpda_single(dt, sensor_para, isSimulation = True)

    # 2. generate observation data and send to the sensor

    # now feed sensor information data into it, emmm, just directly give all informations to them
    time_set = np.linspace(0.1, 50, 500)
    true_target_set = []
    noise_set = []
    total_z = []
    agent0_est = []
    

    for t in time_set:
        true_k, noise_k = generate_obs(t)
        true_target_set.append(true_k)
        noise_set.append(noise_k)
        z_k = true_k + noise_k
        total_z.append(z_k)

        size_k = []
        for i in range(len(z_k)):
            # add observation noice to sensor's measurement
            z_k[i][0] += np.random.normal(0, mtt.R[0,0])
            z_k[i][1] += np.random.normal(0, mtt.R[1,1])
            # size_k is the list of bounding box size for the measurement,
            # here we can just give dummy data for that
            size_k.append([1, 1, 2, 0.1])
        # returns the track result & bounding box infos for all tracks
        
        track_result, bb_output_k = mtt.track_update(t, dt, z_k, size_k) # list of [id, positon]
        

        agent0_est_k = []
        
        for track in track_result:
            x = track.kf.x_k_k[0, 0]
            y = track.kf.x_k_k[1, 0]
            P = track.kf.P_k_k[0:2, 0:2]
            agent0_est_k.append([x, y, P])
        agent0_est.append(agent0_est_k)


    global ax, fig

    # 3. Plot the result
    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-30, 30), ylim=(-30, 30))

    real_point1, = ax.plot([], [], 'bo', ms=2)
    rand_point, = ax.plot([], [], 'go', ms=2)
    real_point1_est, = ax.plot([], [], 'r*', ms=2)
    
    def init():
        """initialize animation"""
        
        real_point1.set_data([], [])
        real_point1_est.set_data([], [])
        
        rand_point.set_data([], [])
        # add FoV
        
        x0 = sensor_para["position"][0]
        y0 = sensor_para["position"][1]
        theta = sensor_para["position"][2]
        h = sensor_para["shape"][1][1]
        w = sensor_para["shape"][1][0]
        # bot left position
        x0 += -0.5 *(- h * np.sin(theta) + w * np.cos(theta))
        y0 += -0.5 *(w * np.sin(theta) + h * np.cos(theta))
        ax.add_artist(Rectangle((x0, y0), 
                    w, h,
                    angle = theta,
                    fc ='none',  
                    ec ='g', 
                    lw = 1,
                    linestyle = '-'))
        
        #     ax.add_artist(sensor_fov_draw_circle)
        return real_point1, rand_point, real_point1_est

    def animate(i):
        """perform animation step"""
        # global ax, fig
        
        # noise observations
        x , y = [], []
        for point in noise_set[i]:
            
            x.append(point[0])
            y.append(point[1])
        rand_point.set_data(x, y)

        x, y = [], []
        for point in true_target_set[i]:
            x.append(point[0])
            y.append(point[1])
        real_point1.set_data(x, y)
        
        #patches = []
        for obj in ax.findobj(match = Ellipse):
            obj.remove()
        
        # print tracks
        x, y = [], []
        for track in agent0_est[i]:
            mu = [track[0], track[1]]
            x.append(mu[0])
            y.append(mu[1])
            S = track[2]
            
            e = make_ellipse(mu, S, mtt.gating_size)
            ax.add_artist(e)
        real_point1_est.set_data(x, y)

        return real_point1, rand_point, real_point1_est

    ani = animation.FuncAnimation(fig, animate, frames=len(time_set),
                                interval=10, blit=True, init_func=init, repeat = False)
    # FFwriter = animation.FFMpegWriter()

    ani.save('video/jpda.mp4', fps=20)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
