import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class ZnModel:
    def __init__(self, lattice_size, n_states, temperature, external_field=0):
        self.L = lattice_size       # Lattice size (L x L)
        self.n = n_states           # Number of states in the Zn model
        self.T = temperature        # Temperature of the system
        self.H = external_field     # External magnetic field
        self.lattice = np.random.randint(0, self.n, (self.L, self.L))  # Initialize lattice randomly
        self.energy = self.calculate_total_energy()
        self.magnetization = self.calculate_magnetization()

    def calculate_total_energy(self):
        total_energy = 0
        for x in range(self.L):
            for y in range(self.L):
                total_energy += self.calculate_energy(x, y)
        return total_energy / 2  # Avoid double counting

    def calculate_energy(self, x, y):
        spin = self.lattice[x, y]
        neighbors = [
            self.lattice[(x + 1) % self.L, y],
            self.lattice[(x - 1) % self.L, y],
            self.lattice[x, (y + 1) % self.L],
            self.lattice[x, (y - 1) % self.L]
        ]
        interaction_energy = -sum(1 if spin == neighbor else 0 for neighbor in neighbors)
        field_energy = -self.H * (1 if spin == 1 else -1)
        return interaction_energy + field_energy

    def calculate_magnetization(self):
        return np.sum(np.array([1 if spin == 1 else -1 for spin in self.lattice.flatten()]))

    def metropolis_step(self):
        for _ in range(self.L * self.L):
            x, y = np.random.randint(0, self.L, size=2)
            current_energy = self.calculate_energy(x, y)

            new_spin = (self.lattice[x, y] + np.random.randint(1, self.n)) % self.n
            old_spin = self.lattice[x, y]
            self.lattice[x, y] = new_spin
            new_energy = self.calculate_energy(x, y)

            delta_energy = new_energy - current_energy
            if delta_energy > 0 and np.random.rand() >= np.exp(-delta_energy / self.T):
                self.lattice[x, y] = old_spin  # Revert if move not accepted
            else:
                self.energy += delta_energy
                self.magnetization += (1 if new_spin == 1 else -1) - (1 if old_spin == 1 else -1)

# Parameters for the animation
lattice_size = 50
n_states = 2
temperature_range = np.linspace(5.0, 0.01, 500)  # gradually change temperature over frames
n_steps = len(temperature_range)

# Instantiate the Zn model with initial temperature
zn_model = ZnModel(lattice_size, n_states, temperature_range[0])

# Set up figure for lattice animation and observables plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im = ax1.imshow(zn_model.lattice, cmap='viridis', vmin=0, vmax=n_states-1)
ax1.set_title("Lattice Configuration")

# Tracking data for observables
energy_data, magnetization_data = [], []
specific_heat_data, susceptibility_data = [], []

# Set up energy and magnetization plots
line_energy, = ax2.plot([], [], label='Energy', color='blue')
line_magnetization, = ax2.plot([], [], label='Magnetization', color='purple')
ax2.set_xlim(0, n_steps)
ax2.set_ylim(-lattice_size**2, lattice_size**2)
ax2.set_title("Energy and Magnetization over Time")
ax2.set_xlabel("Steps")
ax2.legend()

# Initialize the plot
def init():
    line_energy.set_data([], [])
    line_magnetization.set_data([], [])
    return [im, line_energy, line_magnetization]

# Update function for each frame in the animation
def update(frame):
    zn_model.T = temperature_range[frame]  # Update temperature
    zn_model.metropolis_step()
    im.set_array(zn_model.lattice)

    # Update observable data
    energy_data.append(zn_model.energy)
    magnetization_data.append(zn_model.magnetization)
    
    # Calculate specific heat and susceptibility if enough data points are available
    if len(energy_data) > 1:
        # Specific Heat (C) ∝ <E^2> - <E>^2
        specific_heat = (np.var(energy_data) / (lattice_size**2 * zn_model.T**2))
        specific_heat_data.append(specific_heat)
        
        # Susceptibility (χ) ∝ <M^2> - <M>^2
        susceptibility = (np.var(magnetization_data) / (lattice_size**2 * zn_model.T))
        susceptibility_data.append(susceptibility)

    # Update energy and magnetization plots
    line_energy.set_data(range(len(energy_data)), energy_data)
    line_magnetization.set_data(range(len(magnetization_data)), magnetization_data)
    
    # Update titles and limits dynamically
    ax1.set_title(f"Lattice Configuration (T={zn_model.T:.2f})")  # Show current temperature
    ax2.set_ylim(min(energy_data + magnetization_data), max(energy_data + magnetization_data))

    return [im, line_energy, line_magnetization]

# Create the GIF
output_path = "zn_model.gif"
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)
ani.save(output_path, writer=PillowWriter(fps=30))
plt.close()

# Plot results for specific heat and susceptibility after the simulation
plt.figure(figsize=(10, 5))

# Plot specific heat vs temperature
plt.subplot(1, 2, 1)
plt.plot(temperature_range[:len(specific_heat_data)], specific_heat_data, color='red')
plt.title("Specific Heat vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Specific Heat")

# Plot susceptibility vs temperature
plt.subplot(1, 2, 2)
plt.plot(temperature_range[:len(susceptibility_data)], susceptibility_data, color='green')
plt.title("Susceptibility vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")

plt.tight_layout()
plt.show()
