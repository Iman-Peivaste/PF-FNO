#%%
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from tensorflow.keras.utils import to_categorical
#%% Global Parameters

# Physical parameters
dt = 0.005                 # Time step
kappa = 3.0                # Gradient energy coefficient
mobility = 10.0            # PDE mobility parameter
L_domain = 64.0            # Physical domain size (fixed regardless of resolution)
coefA = 1.0                # Coefficient for the gradient term
dexp = -5                  # Exponent parameter 
ntimestep = 8000           # Total number of time steps
nprint = 50                # Save results every nprint steps
num_regions_in = 5         # Expected number of grain regions
seed = 42                  # Random seed

# Input/output settings
input_folder = "Input3"     # Folder containing input PNG images

# Set random seed for reproducibility
np.random.seed(seed)

#%% Data Generation Classes

class GenerateEtas:
    def __init__(self, snapshot, img_dim):
        self.snapshot = snapshot
        self.img_dim = img_dim

    def create_etas(self):
        # Resize input image to desired resolution
        img = cv2.resize(self.snapshot, (self.img_dim, self.img_dim))
        img[img != 255] = 0  # Thresholding: non-255 values become 0
        _, thresholded_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        mask = thresholded_img == 255
        structure = np.ones((3, 3), dtype=int)
        labeled_mask, num_labels = ndimage.label(mask, structure=structure)
        n_classes = len(np.unique(labeled_mask))
        # Drop the background channel (index 0)
        s = to_categorical(labeled_mask, num_classes=n_classes)[:, :, 1:n_classes]
        return s

class ReadInput:
    def __init__(self, dim, input_folder=input_folder):
        self.dim = dim
        self.direct = os.path.join(os.getcwd(), input_folder)

    def read(self):
        if not os.path.exists(self.direct):
            raise FileNotFoundError(f"Directory {self.direct} does not exist.")
        image_files = [f for f in os.listdir(self.direct) if f.endswith('.png')]
        if not image_files:
            raise ValueError(f"No PNG files found in directory {self.direct}.")
        return [cv2.resize(cv2.imread(os.path.join(self.direct, f), 0), (self.dim, self.dim))
                for f in image_files]

class AllenCahnModel:
    def __init__(self, Nx, Ny, num_regions, dx, dy):
        self.Nx, self.Ny, self.num_regions = Nx, Ny, num_regions
        # Build frequency grids taking dx and dy into account
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')

    def k2_k4(self):
        k2 = self.kx**2 + self.ky**2
        return k2**2, k2

    def get_images(self, img_dim):
        reader = ReadInput(dim=img_dim)
        return reader.read()

def free_energy(etas, eta, igrain, ngrain):
    A, B = 1.0, 1.0
    summ = np.sum(etas[:, :, np.arange(ngrain) != igrain]**2, axis=2)
    return A * (2.0 * B * eta * summ + eta**3 - eta)

#%% Main Simulation Function

def run_simulation(resolution):
    """
    Run Allen-Cahn grain growth simulation at specified resolution
    
    Parameters:
    resolution (int): Grid resolution (64, 128, or 256)
    
    Returns:
    tuple: (boundary_data, full_data) - The simulation results
    """
    print(f"\n=== Running simulation at {resolution}×{resolution} resolution ===")
    start_time = time.time()
    
    # Set grid parameters based on resolution
    Nx, Ny = resolution, resolution
    dx = L_domain / Nx
    dy = L_domain / Ny
    
    # Initialize model with resolution and dx, dy
    model = AllenCahnModel(Nx, Ny, num_regions_in, dx, dy)
    all_images = model.get_images(img_dim=resolution)
    
    k4, k2 = model.k2_k4()
    
    # Preallocate arrays for storing results - adjusted for skipping first 4 timesteps
    num_saved_steps = (ntimestep // nprint) - 4  # Skip first 4 save points
    if num_saved_steps <= 0:
        raise ValueError(f"Not enough timesteps to skip first 4. Increase ntimestep or decrease nprint.")
    
    full = np.zeros((len(all_images), num_saved_steps, Nx, Ny), dtype=np.float16)
    boundary = np.zeros_like(full)
    
    for ii, img in tqdm(enumerate(all_images), desc=f'Simulation Progress ({resolution}×{resolution})'):
        try:
            etas = GenerateEtas(img, img_dim=resolution).create_etas()
            num_regions = etas.shape[2]
            glist = np.ones(num_regions)
            
            for step in range(ntimestep):
                for igrain in range(num_regions):
                    if glist[igrain] == 1:
                        eta = etas[:, :, igrain]
                        dfdeta = free_energy(etas, eta, igrain, num_regions)
                        etak = np.fft.fft2(eta)
                        dfdetak = np.fft.fft2(dfdeta)
                        # Use mobility for the PDE update:
                        etak = (etak - dt * mobility * dfdetak) / (1 + dt * coefA * mobility * kappa * k2)
                        eta = np.fft.ifft2(etak).real
                        eta = np.clip(eta, 0.00000, 0.9999)
                        etas[:, :, igrain] = eta
                        
                        if np.mean(eta) < 0.001:
                            glist[igrain] = 0
                            etas[:, :, igrain] = 0
                            continue
                
                # Record data every nprint steps, but skip the first 4 saved steps (which are unstable)
                if step % nprint == 0:
                    save_index = step // nprint
                    # Skip recording the first 4 steps (indices 0-3)
                    if save_index >= 4:
                        adjusted_index = save_index - 4  # Adjust the index for storage
                        if etas.shape[2] > 0:
                            microstructure = np.argmax(etas, axis=2)
                            eta3 = np.sum(etas**2, axis=2)
                        else:
                            microstructure = np.zeros((Nx, Ny), dtype=int)
                            eta3 = np.zeros((Nx, Ny))
                        full[ii, adjusted_index] = microstructure
                        boundary[ii, adjusted_index] = eta3
        except Exception as e:
            print(f"Error processing image {ii} at step {step}: {e}")
            continue
    
    print(f"Simulation at {resolution}×{resolution} completed in {time.time() - start_time:.2f} seconds")
    return boundary, full

#%% Visualization Function

def visualize_example(boundary_data, resolution, sample_idx=0):
    """
    Visualize a single example from the boundary data
    
    Parameters:
    boundary_data: The boundary field data 
    resolution (int): The grid resolution
    sample_idx (int): Which sample to visualize
    """
    n_timepoints = min(5, boundary_data.shape[1])
    fig, axes = plt.subplots(1, n_timepoints, figsize=(n_timepoints*4, 4))
    
    # Select time points at roughly equal intervals
    time_indices = np.linspace(0, boundary_data.shape[1]-1, n_timepoints, dtype=int)
    
    for i, t_idx in enumerate(time_indices):
        axes[i].imshow(boundary_data[sample_idx, t_idx], cmap='viridis')
        # Calculate the actual timestep (accounting for skipped first 4 steps)
        actual_step = (t_idx + 4) * nprint
        axes[i].set_title(f'Step {actual_step}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.suptitle(f'Grain Boundary Evolution at {resolution}×{resolution} Resolution')
    plt.tight_layout()
    plt.savefig(f'example_evolution_{resolution}.png')
    plt.close()

#%% Main Execution - Direct Control

# === CONFIGURATION ===
# Set the resolutions you want to run
resolutions_to_run = [64, 128, 256, 512]  # Change this list to run specific resolutions
#resolutions_to_run = [512]  # Change this list to run specific resolutions
# Set to True to generate visualizations of the evolution
generate_visualizations = True

# Set to True to save the full data (microstructure) in addition to boundary data
save_full_data = False

# === RUN SIMULATIONS ===
overall_start = time.time()

for resolution in resolutions_to_run:
    boundary_data, full_data = run_simulation(resolution)
    
    # Save boundary data
    np.save(f"Dataset2_{resolution}.npy", boundary_data)
    
    # Optionally save full microstructure data
    if save_full_data:
        np.save(f"Dataset_full_{resolution}.npy", full_data)
    
    # Optional visualization
    if generate_visualizations and boundary_data.shape[0] > 0:
        visualize_example(boundary_data, resolution)

print(f"\nAll simulations completed in {time.time() - overall_start:.2f} seconds")
#%%

