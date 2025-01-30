import numba
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = "11"
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# parameters
L = 50.0 # size of box
rho = 2.0 # density
N = int(rho * L**2) # number of particles
print("N:", N)
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 1.0
v0 = 0.5 #r0 / deltat * factor # velocity
iterations = 500 # animation frames
# eta = 0.2 # noise/randomness
eta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
max_neighbours = N // 2 #  guess a good value, max is N

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
# angles = np.pi/2*np.ones(N) 
angles = np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

# cell list
cell_size = 1.0 * r0
lateral_num_cells = int(L / cell_size)
total_num_cells = lateral_num_cells ** 2
max_particles_per_cell = int(rho * cell_size ** 2 * 10)

# average angles
time_step = 10
frames_time_step = np.empty(time_step)
t = 0
average_angles = [] # empty array for average angles
alignment_data = {} # dictionary for aligment data for each eta
order_parameters = []

# histogram for average particle density in different areas of the box
bins = int(L / (r0/2))
hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, density = False)

threshold = r0
all_num_clusters = [] # empty array for number of clusters
all_avg_cluster_particles = [] # empty array for average num of particles per cluster

@numba.njit()
def get_cell_index(pos, cell_size, num_cells):
    return int(pos[0] // cell_size) % num_cells, int(pos[1] // cell_size) % num_cells

@numba.njit(parallel=True)
def initialize_cells(positions, cell_size, num_cells, max_particles_per_cell):
    
    # create cell arrays
    cells = np.full((num_cells, num_cells, max_particles_per_cell), -1, dtype = np.int32)  # -1 means empty
    cell_counts = np.zeros((num_cells, num_cells), dtype = np.int32)
    
    # populate cells with particle indices
    for i in numba.prange(positions.shape[0]):
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)
        idx = cell_counts[cell_x, cell_y]
        if idx < max_particles_per_cell:
            cells[cell_x, cell_y, idx] = i  # add particle index to cell
            cell_counts[cell_x, cell_y] += 1  # update particle count in this cell
    return cells, cell_counts

@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles * 1.0j)))

average_angles = [average_angle(positions)]

@numba.njit
def order_parameter(angles):
    avg_velocity = np.array([np.cos(angles), np.sin(angles)]).mean(axis = 1)
    order_param = np.linalg.norm(avg_velocity)
    # order_parameters.append(order_param)
    return order_param

def clusters(positions, L, threshold):
    # taking into account periodic boundary conditions
    total = 0
    for d in range(positions.shape[1]):
        pd = pdist(positions[:, d].reshape(positions.shape[0],1))
        pd[pd > L * 0.5] -= L
        total += pd ** 2
    total = np.sqrt(total)
    square = squareform(total)
    
    # clustering
    clustering = DBSCAN(eps = threshold, metric = "precomputed").fit(square)
    labels = clustering.labels_ # assign cluster labels to each point (points labelled -1 are noise)
    unique_labels = set(labels) # unique clustering labels
    
    # exclude noise in number of clusters calculation
    if -1 in labels:
        num_clusters = len(unique_labels) - 1
    else:
        num_clusters = len(unique_labels)
        
    # average number of particles per cluster
    if num_clusters > 0:
        avg_cluster_particles = len(positions) / num_clusters
    else:
        avg_cluster_particles = 0
        
    return num_clusters, avg_cluster_particles      

@numba.njit(parallel=True)
def update(positions, angles, cell_size, num_cells, max_particles_per_cell, eta):
    
    # empty arrays to hold updated positions and angles
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
      
    # initialize cell lists
    cells, cell_counts = initialize_cells(positions, cell_size, num_cells, max_particles_per_cell)

    # loop over all particles
    for i in numba.prange(N):  # parallelize outer loop
        neighbour_angles = np.empty(max_neighbours)
        count_neighbour = 0

        # get particle's cell, ensuring indices are integers
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)

        # check neighboring cells (3x3 neighborhood)
        for cell_dx in (-1, 0, 1):
            for cell_dy in (-1, 0, 1):
                # ensure neighbor_x and neighbor_y are integers
                neighbour_x = int((cell_x + cell_dx) % num_cells)
                neighbour_y = int((cell_y + cell_dy) % num_cells)

                # check each particle in the neighboring cell
                for idx in range(cell_counts[neighbour_x, neighbour_y]):
                    j = cells[neighbour_x, neighbour_y, idx]
                    if i != j:  # avoid self-comparison
                        # calculate squared distance for efficiency
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance_sq = dx * dx + dy * dy
                        # compare with squared radius
                        if distance_sq < r0 * r0:
                            if count_neighbour < max_neighbours:
                                neighbour_angles[count_neighbour] = angles[j]
                                count_neighbour += 1

        # apply noise using Numba-compatible randomness
        noise = eta * (np.random.random() * 2 * np.pi - np.pi)

        # if neighbours, calculate average angle
        if count_neighbour > 0:
            average_angle = np.angle(np.sum(np.exp(neighbour_angles[:count_neighbour] * 1j)))
            new_angles[i] = average_angle + noise
        else:
            # if no neighbours, keep current angle
            new_angles[i] = angles[i] + noise

        # update position based on new angle
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # apply boundary conditions of box
        new_positions[i] %= L

    return new_positions, new_angles

def animate(frames):
    print(frames)
    global positions, angles, frames_time_step, t, hist
    
    new_positions, new_angles = update(positions, angles, cell_size, lateral_num_cells, max_particles_per_cell, eta)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # update the empty array with average angle
    frames_time_step[t] = average_angle(new_angles)
    if t == time_step - 1:  # check if array filled
        average_angles.append(average_angle(frames_time_step))
        t = 0  # reset t
        frames_time_step = np.empty(time_step)  # reinitialise the array
    else:
        t += 1  # increment t
    
    # add particle positions to the histogram
    hist += np.histogram2d(positions[:,0], positions[:,1], bins = [xedges, yedges], density = False)[0]
    
    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    np.savez_compressed(f"pos_ang_arrays/bands/frame{frames}.npz", positions = np.array(positions, dtype = np.float16), angles = np.array(angles, dtype = np.float16))
    return qv,

# Alignment of Particles for Different Noise
for eta in eta_values:
    # reset positions and angles for each eta
    positions = np.random.uniform(0, L, size = (N, 2))
    angles = np.random.uniform(-np.pi, np.pi, size = N)
    
    average_angles = [] # initialise average angles array
    
    hist = np.empty((len(xedges) - 1, len(yedges) - 1)) # initialise histogram density map
    
    # Vicsek Model for N Particles Animation
    fig, ax = plt.subplots(figsize = (3.5, 3.5)) 
    qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
    anim = FuncAnimation(fig, animate, frames = range(0, iterations + 1), interval = 5, blit = True)
    # writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
    # anim.save("Vicsek_bands.mp4", writer = writer, dpi = 300)
    # plt.show()
    
    # First and Last Frame Vicsek Model for N particles
    start_positions = positions.copy()
    start_angles = angles.copy()

    for frame in range(0, iterations + 1):
        animate(frame)
        
        # store alignment for each eta    
        alignment_data[eta] = average_angles
         
    end_positions = positions
    end_angles = angles
    
    num_clusters, avg_cluster_particles = clusters(positions, L, threshold)
    all_num_clusters.append(num_clusters)
    all_avg_cluster_particles.append(avg_cluster_particles)

    fig, (ax4, ax5) = plt.subplots(1, 2, figsize = (7, 3))
    ax4.set_aspect("equal")
    ax4.quiver(start_positions[:,0], start_positions[:,1], np.cos(start_angles), np.sin(start_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
    ax4.set_title("Frame 0")
    ax4.set_xlabel(f"Noise = {eta}")
    ax4.set_xticks(range(0, 51, 10))
    ax4.set_yticks(range(0, 51, 10))
    ax5.set_aspect("equal")
    ax5.quiver(end_positions[:,0], end_positions[:,1], np.cos(end_angles), np.sin(end_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
    ax5.set_title(f"Frame {iterations}")
    ax5.set_xlabel(f"$\eta$ = {eta}")
    ax5.set_xticks(range(0, 51, 10))
    ax5.set_yticks(range(0, 51, 10))
    plt.tight_layout()
    # plt.savefig(f"Vicsek_bands_14_{int(eta*10)}.png", dpi = 300)
    # plt.show()
    
    # normalise the histogram to cartesian coordinates for plotting
    hist_normalised = hist.T / np.sum(hist)

    # Normalised 2D Histogram of Particle Density
    fig, ax3 = plt.subplots(figsize = (3.5, 2.5))
    cax = ax3.imshow(hist_normalised, extent = [0, L, 0, L], origin = "lower", cmap = "hot", aspect = "auto")
    ax3.set_xticks(range(0, 51, 10))
    ax3.set_yticks(range(0, 51, 10))
    ax3.set_title(f"$\eta$ = {eta}")
    fig.colorbar(cax, ax = ax3, label = "Density")
    plt.tight_layout()
    # plt.savefig(f"densitymap_14_{int(eta*10)}.png", dpi = 300)
    # plt.show()
    
fig, ax6 = plt.subplots(figsize = (3.5, 2.5))
for eta, avg_angles in alignment_data.items(): # plot average angles for each eta
    times = np.arange(0, len(avg_angles)) * time_step
    ax6.plot(times, avg_angles, label = f"$\eta$ = {eta}") 
ax6.set_xlabel("Time Step")
ax6.set_ylabel("Average Angle (radians)")
ax6.set_xticks(range(0, 501, 100))
ax6.legend()
plt.tight_layout()
# plt.savefig("alignment_eta_14.png")
# plt.show()

fig, (ax7, ax8) = plt.subplots(1, 2, figsize = (7, 3))
ax7.plot(eta_values, all_num_clusters, marker = ".")
ax7.set_xticks(np.arange(0.1, 0.51, 0.1))
ax7.set_yticks(range(0, int(max(all_num_clusters)+1)), 10)
ax7.set_xlabel("Noise")
ax7.set_ylabel("Number of Clusters")
ax8.plot(eta_values, all_avg_cluster_particles, marker = ".")
ax8.set_xticks(np.arange(0.1, 0.51, 0.1))
ax8.set_yticks(range(0, int(max(all_avg_cluster_particles)+1), 50))
ax8.set_xlabel("Noise")
ax8.set_ylabel("Average Number of Particles\n per Cluster")
plt.tight_layout()
# plt.savefig("clusters_15.png")
plt.show()

# # Vicsek Model for N Particles Animation
# fig, ax = plt.subplots(figsize = (3.5, 3.5)) 
# qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# anim = FuncAnimation(fig, animate, frames = range(0, iterations), interval = 5, blit = True)
# writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
# # anim.save("Vicsek_bands.mp4", writer = writer, dpi = 300)
# plt.show()

# # First and Last Frame Vicsek Model for N particles
# start_positions = positions.copy()
# start_angles = angles.copy()

# for frame in range(0, iterations + 1):
#     animate(frame)
    
# end_positions = positions
# end_angles = angles

# fig, (ax4, ax5) = plt.subplots(1, 2, figsize = (7, 3))
# ax4.set_aspect("equal")
# ax4.quiver(start_positions[:,0], start_positions[:,1], np.cos(start_angles), np.sin(start_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# ax4.set_title("Frame 0")
# ax4.set_xticks(range(0, 51, 10))
# ax4.set_yticks(range(0, 51, 10))
# ax5.set_aspect("equal")
# ax5.quiver(end_positions[:,0], end_positions[:,1], np.cos(end_angles), np.sin(end_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# ax5.set_title(f"Frame {iterations}")
# ax5.set_xticks(range(0, 51, 10))
# ax5.set_yticks(range(0, 51, 10))
# plt.tight_layout()
# # plt.savefig("Vicsek_bands_9.png", dpi = 300)
# plt.show()

# # Alignment of Particles over Time
# fig, ax2 = plt.subplots(figsize = (3.5, 2.5))
# times = np.arange(0,len(average_angles))*time_step
# ax2.plot(times, average_angles)
# ax2.set_xlabel("Time Step")
# ax2.set_ylabel("Average Angle (radians)")
# ax2.set_xticks(range(0, 501, 100))
# plt.tight_layout()
# # plt.savefig("alignment_9.png", dpi = 300)
# plt.show()

# # normalise the histogram to cartesian coordinates for plotting
# hist_normalised = hist.T / sum(hist)

# # Normalised 2D Histogram of Particle Density
# fig, ax3 = plt.subplots(figsize = (3.5, 2.5))
# cax = ax3.imshow(hist_normalised, extent = [0, L, 0, L], origin = "lower", cmap = "hot", aspect = "auto")
# ax3.set_xticks(range(0, 51, 10))
# ax3.set_yticks(range(0, 51, 10))
# fig.colorbar(cax, ax = ax3, label = "Density")
# plt.tight_layout()
# # plt.savefig("densitymap_9.png", dpi = 300)
# plt.show()