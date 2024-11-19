import os
import time
import numpy as np
import cv2
from tqdm import tqdm
from scipy import ndimage
from tensorflow.keras.utils import to_categorical


# Utility Constants
IMG_DIM = 64
DT = 0.005
KAPPA = 3.0
L = 10
COEF_A = 1.0
NPRINT = 50
NTIMESTEP = 8000
SEED = 42

np.random.seed(SEED)


class ImageReader:
    def __init__(self, input_folder="Input", dim=IMG_DIM):
        self.dim = dim
        self.input_folder = os.path.join(os.getcwd(), input_folder)

    def read_images(self):
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError(f"Directory {self.input_folder} does not exist.")
        image_files = [f for f in os.listdir(self.input_folder) if f.endswith('.png')]
        if not image_files:
            raise ValueError(f"No PNG files found in directory {self.input_folder}.")
        return [cv2.resize(cv2.imread(os.path.join(self.input_folder, f), 0), (self.dim, self.dim)) for f in image_files]


class GrainStructure:
    def __init__(self, image, img_dim=IMG_DIM):
        self.image = cv2.resize(image, (img_dim, img_dim))

    def generate_etas(self):
        self.image[self.image != 255] = 0  # Thresholding
        _, thresholded = cv2.threshold(self.image, 200, 255, cv2.THRESH_BINARY)
        mask = thresholded == 255
        structure = np.ones((3, 3), dtype=int)
        labeled_mask, _ = ndimage.label(mask, structure=structure)
        n_classes = len(np.unique(labeled_mask))
        return to_categorical(labeled_mask, num_classes=n_classes)[:, :, 1:]


class AllenCahnSimulator:
    def __init__(self, Nx, Ny, dx, dy, num_regions):
        self.Nx, self.Ny, self.num_regions = Nx, Ny, num_regions
        self.kx, self.ky = self._initialize_wavenumbers(dx, dy)

    def _initialize_wavenumbers(self, dx, dy):
        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=dy)
        return np.meshgrid(kx, ky, indexing='ij')

    def compute_k2_k4(self):
        k2 = self.kx**2 + self.ky**2
        return k2**2, k2

    def run_simulation(self, images):
        k4, k2 = self.compute_k2_k4()
        full_data = np.zeros((len(images), NTIMESTEP // NPRINT, self.Nx, self.Ny), dtype=np.float16)
        boundary_data = np.zeros_like(full_data)

        for ii, image in tqdm(enumerate(images), desc="Simulation Progress"):
            try:
                etas = GrainStructure(image).generate_etas()
                num_regions = etas.shape[2]
                glist = np.ones(num_regions)
                elapsed_time = 0

                for step in range(NTIMESTEP):
                    elapsed_time += DT
                    for igrain in range(num_regions):
                        if glist[igrain] == 1:
                            eta = etas[:, :, igrain]
                            dfdeta = self._compute_free_energy(etas, eta, igrain, num_regions)
                            etak = np.fft.fft2(eta)
                            dfdetak = np.fft.fft2(dfdeta)
                            etak = (etak - DT * L * dfdetak) / (1 + DT * COEF_A * L * KAPPA * k2)
                            eta = np.fft.ifft2(etak).real
                            eta = np.clip(eta, 0.0, 1.0)
                            etas[:, :, igrain] = eta

                            if np.mean(eta) < 0.001:
                                glist[igrain] = 0
                                etas[:, :, igrain] = 0

                    if step % NPRINT == 0:
                        microstructure, eta3 = self._extract_microstructure(etas)
                        full_data[ii, step // NPRINT] = microstructure
                        boundary_data[ii, step // NPRINT] = eta3

            except Exception as e:
                print(f"Error processing image {ii}: {e}")
                continue

        return full_data, boundary_data

    @staticmethod
    def _compute_free_energy(etas, eta, igrain, ngrain):
        A, B = 1.0, 1.0
        summ = np.sum(etas[:, :, np.arange(ngrain) != igrain]**2, axis=2)
        return A * (2.0 * B * eta * summ + eta**3 - eta)

    @staticmethod
    def _extract_microstructure(etas):
        if etas.shape[2] > 0:
            microstructure = np.argmax(etas, axis=2)
            eta3 = np.sum(etas**2, axis=2)
        else:
            microstructure = np.zeros((etas.shape[0], etas.shape[1]), dtype=int)
            eta3 = np.zeros_like(microstructure)
        return microstructure, eta3


if __name__ == "__main__":
    start_time = time.time()

    # Read input images
    reader = ImageReader()
    images = reader.read_images()

    # Initialize simulator
    simulator = AllenCahnSimulator(Nx=64, Ny=64, dx=1.0, dy=1.0, num_regions=5)

    # Run simulation
    full_data, boundary_data = simulator.run_simulation(images)

    # Save results
    np.save("Dataset_64.npy", boundary_data)

    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds")
