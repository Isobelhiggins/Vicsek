import numba
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = "11"
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
eta = 0.2 # noise/randomness
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

# histogram for average particle density in different areas of the box
bins = int(L / (r0/2))
hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, density = False)

tot_vx_all = np.zeros((bins, bins))
tot_vy_all = np.zeros((bins, bins))
counts_all = np.zeros((bins, bins))
vxedges = np.linspace(0, 1, bins + 1)
vyedges = np.linspace(0, 1, bins + 1)

# define barrier position and size
barrier_x_start, barrier_x_end = 15, 35
barrier_y_start, barrier_y_end = 15, 35

# barrier collision avoidance
turnfactor = np.pi/2
boundary = r0
turned = np.zeros(N, dtype = bool)
    
@numba.njit()
def barrier_collision(position, angle, turned):
    # calculate next position based on current position and angle
    next_position = (position + v0 * np.array([np.cos(angle), np.sin(angle)]) * deltat)
    next_angle = angle
    
    # if particle has not turned
    if turned == False:
            
        if ((next_position[0] <= boundary or next_position[0] >= (L - boundary)) or 
            (next_position[1] <= boundary or next_position[1] >= (L - boundary))):
            next_angle += turnfactor
            turned = True
            
        next_position = (position + v0 * np.array([np.cos(next_angle), np.sin(next_angle)]) * deltat)
    
    if ((next_position[0] <= 0 or next_position[0] >= L) or
        (next_position[1] <= 0 or next_position[1] >= L)):
        next_angle += turnfactor
        turned = True
        return position, next_angle, turned
        
    return next_position, next_angle, turned

@numba.njit()
def get_cell_index(pos, cell_size, num_cells):
    return int(pos[0] // cell_size) % num_cells, int(pos[1] // cell_size) % num_cells

@numba.njit(parallel=True)
def initialize_cells(positions, cell_size, num_cells, max_particles_per_cell):
    
    # create cell arrays
    cells = np.full((num_cells, num_cells, max_particles_per_cell), -1, dtype = np.int32)  # -1 means empty
    cell_counts = np.zeros((num_cells, num_cells), dtype = np.int32)
    
    # populate cells with particle indices
    for i in  numba.prange(positions.shape[0]):
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
def velocity_flux(positions, angles, v0):
    
    tot_vx = np.zeros(positions.shape[0])
    tot_vy = np.zeros(positions.shape[0])
    
    for i in numba.prange(positions.shape[0]):
        tot_vx[i] = v0 * np.cos(angles[i])
        tot_vy[i] = v0 * np.sin(angles[i])
    
    return tot_vx, tot_vy

@numba.njit(parallel=True)
def update(positions, angles, turned, cell_size, num_cells, max_particles_per_cell):
    
    # empty arrays to hold updated positions and angles
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    new_turned = np.empty_like(turned)
      
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
        
        # update position with barrier collision check
        new_positions[i], new_angles[i], new_turned[i] = barrier_collision(positions[i], new_angles[i], turned[i])
        # # boundary conditions of box
        # new_positions[i] %= L
        
        # # boundary conditions (particles constrained to box)
        # if (new_positions[i, 0] < 0 or new_positions[i, 0] > L) or (new_positions[i, 1] < 0 or new_positions[i, 1] > L):
        #     new_positions[i] = positions[i] # do not update position if out of bounds

    return new_positions, new_angles, new_turned

def animate(frames):
    print(frames)
    global positions, angles, turned, frames_time_step, t, hist, tot_vx_all, tot_vy_all, counts_all, vxedges, vyedges
    
    new_positions, new_angles, new_turned = update(positions, angles, turned, cell_size, lateral_num_cells, max_particles_per_cell)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    turned = new_turned
    
    # update the empty array with average angle
    frames_time_step[t] = average_angle(new_angles)
    if t == time_step - 1:  # check if array filled
        average_angles.append(average_angle(frames_time_step))
        t = 0  # reset t
        frames_time_step = np.empty(time_step)  # reinitialise array
    else:
        t += 1  # increment t
    
    # add particle positions to the histogram
    hist += np.histogram2d(positions[:,0], positions[:,1], bins = [xedges, yedges], density = False)[0]
    
    tot_vx, tot_vy = velocity_flux(positions, angles, v0)
    
    H_vx, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, weights = tot_vx)
    H_vy, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, weights = tot_vy)
    counts, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins)
    
    tot_vx_all += H_vx
    tot_vy_all += H_vy
    counts_all += counts
    
    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    np.savez_compressed(f"pos_ang_arrays/bands_repulsive/frame{frames}.npz", positions = np.array(positions, dtype = np.float16), angles = np.array(angles, dtype = np.float16))
    return qv,

# Vicsek Model for N Particles Animation
fig, ax = plt.subplots(figsize = (3.5, 3.5)) 
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax.add_patch(plt.Rectangle((0, 0), L, L, edgecolor = "grey", fill = False)) # box
ax.add_patch(plt.Rectangle((boundary, boundary), (L - 2*boundary), (L - 2*boundary), edgecolor = "grey", fill = False)) # box boundary
anim = FuncAnimation(fig, animate, frames = range(0, iterations), interval = 5, blit = True)
writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
# anim.save("Vicsek_bands_trapped.mp4", writer = writer, dpi = 300)
# plt.show()

# First and Last Frame Vicsek Model for N particles
# start_positions = positions.copy()
# start_angles = angles.copy()

# for frame in range(0, iterations + 1):
#     animate(frame)
    
# end_positions = positions
# end_angles = angles

# fig, (ax4, ax5) = plt.subplots(1, 2, figsize = (7, 3))
# ax4.set_aspect("equal")
# ax4.quiver(start_positions[:,0], start_positions[:,1], np.cos(start_angles), np.sin(start_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# ax4.add_patch(plt.Rectangle((barrier_x_start, barrier_y_start), barrier_x_end - barrier_x_start, barrier_y_end - barrier_y_start, color = "grey", alpha = 0.5))
# ax4.set_title("Frame 0")
# ax4.set_xticks(range(0, 51, 10))
# ax4.set_yticks(range(0, 51, 10))
# ax5.set_aspect("equal")
# ax5.quiver(end_positions[:,0], end_positions[:,1], np.cos(end_angles), np.sin(end_angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# ax5.add_patch(plt.Rectangle((barrier_x_start, barrier_y_start), barrier_x_end - barrier_x_start, barrier_y_end - barrier_y_start, color = "grey", alpha = 0.5))
# ax5.set_title(f"Frame {iterations}")
# ax5.set_xticks(range(0, 51, 10))
# ax5.set_yticks(range(0, 51, 10))
# plt.tight_layout()
# # plt.savefig("Vicsek_bands_repulsive_14.png", dpi = 300)
# plt.show()


# Alignment of Particles over Time
fig, ax2 = plt.subplots(figsize = (3.5, 2.5))
times = np.arange(0,len(average_angles))*time_step
ax2.plot(times, average_angles)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Average Angle (radians)")
ax2.set_xticks(range(0, 501, 100))
plt.tight_layout()
# plt.savefig("repulsive_alignment_14.png", dpi = 300)
# plt.show()

# normalise the histogram to cartesian coordinates for plotting
hist_normalised = hist.T / sum(hist)

# Normalised 2D Histogram of Particle Density
fig, ax3 = plt.subplots(figsize = (3.5, 2.5))
cax = ax3.imshow(hist_normalised, extent = [0, L, 0, L], origin = "lower", cmap = "hot", aspect = "auto")
ax3.set_xticks(range(0, 51, 10))
ax3.set_yticks(range(0, 51, 10))
fig.colorbar(cax, ax = ax3, label = "Density")
plt.tight_layout()
# plt.savefig("repulsive_densitymap_14.png", dpi = 300)
# plt.show()

avg_vx = np.zeros_like(tot_vx_all)
avg_vy = np.zeros_like(tot_vy_all)
avg_vx[counts_all > 0] = tot_vx_all[counts_all > 0] / counts_all[counts_all > 0]
avg_vy[counts_all > 0] = tot_vy_all[counts_all > 0] / counts_all[counts_all > 0]

X, Y = np.meshgrid(vxedges[:-1], vyedges[:-1])

fig, ax6 = plt.subplots(figsize = (3.5, 3.5))
ax6.quiver(X, Y, avg_vx.T, avg_vy.T)
plt.tight_layout()
plt.show()
