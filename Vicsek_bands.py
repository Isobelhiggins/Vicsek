import numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# parameters
L = 100.0 # size of box
rho = 0.5 # density
N = int(rho * L**2) # number of particles
print("N:", N)
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 1.
v0 = 0.5 #r0 / deltat * factor # velocity
iterations = 100 # animation frames
eta = 0.3 # noise/randomness
max_neighbours = N // 2 #  guess a good value, max is N

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.pi/2*np.ones(N) #np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

# cell list
cell_size = 1.0 * r0
lateral_num_cells = int(L / cell_size)
total_num_cells = lateral_num_cells ** 2
max_particles_per_cell = int(rho * cell_size ** 2 * 10)

# empty array for average angles of particles every 10 time steps
average_angles = np.empty(iterations // 10)

# histogram for average particle density in different areas of the box
bins = int(L / (r0/2))
hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, density = False)

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

@numba.njit(parallel=True)
def update(positions, angles, cell_size, num_cells, max_particles_per_cell):
    
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
    global positions, angles, hist
    
    new_positions, new_angles = update(positions, angles, cell_size, lateral_num_cells, max_particles_per_cell)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # update the empty array with average angle
    average_angles[frames // 10] = np.angle(np.mean(np.exp(angles * 1j)))
    
    # add particle positions to the histogram
    hist += np.histogram2d(positions[:,0], positions[:,1], bins = [xedges, yedges], density = False)[0]
    
    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    np.savez_compressed(f"trajectory/frame{frames}.npz", positions=np.array(positions,dtype=np.float16), angles=np.array(angles,dtype=np.float16))
    return qv,

fig, ax = plt.subplots(figsize = (6, 6)) 

qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax.set_title(f"Vicsek model for {N} particles")
# ax.grid()
# for i in range(10000):
#     animate(i)
anim = FuncAnimation(fig, animate,  interval = 5, blit = True)
writer = FFMpegWriter(fps = 10, metadata = dict(artist = "Isobel"), bitrate = 1800)
#anim.save("Vicsek_loops.mp4", writer = writer, dpi = 100)
plt.show()

fig, ax2 = plt.subplots(figsize = (6, 6))
ax2.plot(range(0, iterations, 10), average_angles)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Average Angle (radians)")
ax2.set_title("Alignment of Particles over Time")
#plt.savefig("alignment.png")
plt.show()

# normalise the histogram to cartesian coordinates for plotting
hist_normalised = hist.T / sum(hist)

fig, ax3 = plt.subplots(figsize = (12, 12))
cax = ax3.imshow(hist_normalised, extent = [0, L, 0, L], origin = "lower", cmap = "hot", aspect = "auto")
ax3.set_xlabel("X Position")
ax3.set_ylabel("Y Position")
ax3.set_title("Normalised 2D Histogram of Particle Density")
fig.colorbar(cax, ax = ax3, label = "Density")
#plt.savefig("densitymap.png")
plt.show()