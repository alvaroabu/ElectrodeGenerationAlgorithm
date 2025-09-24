from typing import List
import time
import porespy as ps
import numpy as np
import numpy.typing as npt
from numba import njit, stencil
import h5py
import random
import itertools

from signals import gauss, lognormal
from porespy.tools import _make_ellipsoid

class Structure:
    def __init__(self, box: List[float], voxel: float):
        """
        Initialize the Structure class.

        Parameters:
        - box: List of the dimensions of the 3D box in the form [x, y, z].
        - voxel: Size of a single voxel.
        """
        # Convert box dimensions to numpy array and set the voxel size.
        self.box = np.array(box, dtype=float)
        self.voxel = voxel
        
        # Calculate the shape of the 3D grid by dividing the box dimensions by voxel size.
        self.box_shape = (self.box / self.voxel).astype(int)
        
        # Initialize a 3D numpy array to represent the structure, filled with zeros.
        self.im = np.zeros(self.box_shape, dtype=np.uint16)

    @staticmethod
    def calculate_porosity(array: npt.NDArray[np.uint16]):
        """
        Calculate the porosity of a 3D numpy array.
        
         Parameters:
         - array: Input 3D numpy array.
        
         Returns:
         - Porosity as a float value.
         """
        # Count the number of non-zero (solid) elements.
        solid_num = np.count_nonzero(array)
        # Calculate the total number of elements.
        total_num = array.size

        # Return the porosity: 1 - (solid volume fraction).
        return 1.0 - solid_num / total_num

    @staticmethod
    def vol_frac(array: npt.NDArray[np.uint16]):
        """
        Calculate the volume fraction of solid material in the 3D array.

        Parameters:
        - array: Input 3D numpy array.

        Returns:
        - Volume fraction as a float value.
        """
        # Count the number of non-zero (solid) elements.
        solid_num = np.count_nonzero(array)
        # Calculate the total number of elements.
        total_num = array.size

        # Return the volume fraction of solid material.
        return solid_num / total_num

    @staticmethod
    def calculate_voxel_num(array: npt.NDArray[np.uint16], vol_frac: float):
        """
        Calculate the number of solid voxels required to achieve a given volume fraction.

        Parameters:
        - array: Input 3D numpy array.
        - vol_frac: Desired volume fraction as a float.

        Returns:
        - Number of solid voxels as an integer.
        """
        # Calculate the total number of elements.
        total_num = array.size

        # Calculate and return the number of solid voxels based on the desired volume fraction.
        return int(total_num * vol_frac + 0.5)

    def to_vtk(self, filename: str):
        """
        Export the 3D structure to a VTK file.

        Parameters:
        - filename: Output filename for the VTK file.
        """
        ps.io.to_vtk(self.im, filename=filename)

    def to_hdf5(self, filename: str): # Manually add the file extension 
        """
        Save the structure to an HDF5 file.

        Parameters:
        - filename: Output filename for the HDF5 file.
        """
        # Create a new HDF5 file.
        dataset = h5py.File(filename, 'w')
        
        # Store the image data and box dimensions in the HDF5 file.
        dataset.create_dataset('im', data=self.im)
        dataset.create_dataset('box', data=self.box)
        
        # Close the file after saving.
        dataset.close()

    @classmethod
    def from_hdf5(cls, filename: str):
        """
        Load a structure from an HDF5 file.

        Parameters:
        - filename: Input filename for the HDF5 file.

        Returns:
        - An instance of the Structure class.
        """
        # Open the HDF5 file in read mode.
        dataset = h5py.File(filename, 'r')
        
        # Load the image data and box dimensions.
        im = dataset['im'][:]
        box_shape = im.shape
        box = dataset['box'][:]
        
        # Calculate the voxel size from the box dimensions and image shape.
        voxel = box[0] / box_shape[0]
        
        # Close the file after reading.
        dataset.close()

        # Create a new Structure instance with the loaded data.
        obj = cls(box=box, voxel=voxel)
        obj.im = im
        return obj
   
@staticmethod
def cut_electrode(elec_to_cut: np.ndarray, last_diam: float, axis: str) -> np.ndarray:
    """
    Cut an electrode structure along a specified axis by removing empty sections.

    Parameters:
    - elec_to_cut: 3D numpy array representing the electrode structure.
    - last_diam: Diameter of the electrode feature to be considered for cutting.
    - axis: Axis along which the cutting should be performed ('x', 'y', or 'z').

    Returns:
    - Modified electrode structure after cutting.
    """
    # Map axis letters to array dimensions and slicing details.
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("Incorrect axis. Must be 'x', 'y', or 'z'.")

    # Get the axis index based on the axis letter.
    axis_index = axis_map[axis]
    
    # Calculate the length of the specified axis.
    axis_length = elec_to_cut.shape[axis_index]

    # Define the reduction factor based on last_diam.
    reduction_factor = 0.67 if last_diam >= 30 else 0.50 if last_diam >= 20 else 0.33

    # Traverse through the structure along the specified axis.
    for i in range(axis_length):
        # Slice along the axis to check for empty regions.
        sliced_array = np.take(elec_to_cut, indices=range(i, axis_length), axis=axis_index)
        
        # If no non-zero elements are found in the remaining array along the axis, perform cutting.
        if np.count_nonzero(sliced_array) == 0:
            # Calculate the range to be cut based on `reduction_factor`.
            cut_start = max(0, i - int(reduction_factor * last_diam))
            elec_to_cut = np.delete(elec_to_cut, np.arange(cut_start, axis_length), axis=axis_index)
            break

    return elec_to_cut

            
class StructureSpheres(Structure):

    def __init__(self, box: List[float] = None, voxel: float = None):
        """
        Initialize a StructureSpheres object, inheriting properties from the Structure class.
        
        Parameters:
        - box: List of dimensions for the 3D box structure.
        - voxel: Size of a single voxel in the structure.
        """
        super().__init__(box=box, voxel=voxel)  # Call the parent class initializer.

        # Dictionary to store various statistics about the structure.
        self.statistics = dict()
        self.statistics['porosity'] = 1.0  # Default porosity is set to 1.0 (fully porous).
        self.statistics['diameters'] = {'Total': []}  # Stores diameters of spheres.
        self.statistics['angles'] = dict()  # Placeholder for angle information.
        self.data = {'id':[], 'x': [], 'y': [], 'z': [], 'r': []}

    def _estimate_number(self, target_porosity: float,
                        diams: npt.NDArray[np.float64],
                        cfd: npt.NDArray[np.float64],
                        ratios: List[float] = None,
                        particle: str = 'sphere'):
        
        """ Estimate number of particles to respect volume ratios """
        particle_diameters = []
        volumes = []
        targetvol = self.im.size * (self.voxel ** 3) * (1.0 - target_porosity) * 1.8
        if ratios is None:
            ratios = [1.0]

        for i, ratio in enumerate(ratios):
            volume = 0.0
            diameters = []
            while volume < targetvol * ratio:
                rnd = np.random.random_sample()
                idmin = (np.abs(cfd[i] - rnd)).argmin()
                if particle == 'sphere':
                    diam = diams[idmin]
                    volume += np.pi * diam ** 3 / 6.0
                    diameters.append(diam)
                elif particle == 'ellipsoid':
                    diams_x = diams[0][idmin]
                    diams_y = diams[1][idmin]
                    diams_z = diams[2][idmin]
                    volume += np.pi * diams_x * diams_y * diams_z / 6.0
                    diameters.append([diams_x, diams_y, diams_z])

            volumes.append(volume)
            particle_diameters.append(diameters)

        print(" Objective volume ratios", ratios)
        print(" Extracted volume ratios", [vol/sum(volumes) for vol in volumes])
        print(" Number of particles", [len(parts) for parts in particle_diameters])

        return particle_diameters    

    def _estimate_count(self, target_porosity: float,
                        diams: npt.NDArray[np.float64],
                        cfd: npt.NDArray[np.float64],
                        ratios: List[float] = None,
                        geometry: str = 'sphere'):
       """
       Estimate the number of spheres needed to achieve the target porosity.

       Parameters:
       - target_porosity: Desired porosity level of the structure.
       - diams: Array of diameters for the spheres.
       - cfd: Cumulative frequency distribution array.
       - ratios: Ratios of different sphere sizes.
       - geometry: Shape of the elements (default is 'sphere').

       Returns:
       - A list containing the estimated number of spheres for each size group.
       """
       count = []
       # Calculate target volume based on porosity and total volume of the image.
       targetvol = self.im.size * (self.voxel ** 3) * (1.0 - target_porosity)

       # If specific ratios are provided, estimate counts for each ratio group.
       if ratios is not None:
           for i, ratio in enumerate(ratios):
               # Calculate average volume based on geometry.
               if geometry == 'sphere':
                   diam_avg = diams.mean()  # Average diameter for a sphere.
                   vol = np.pi * diam_avg ** 3 / 6.0  # Volume of a sphere.
               elif geometry == 'ellipsoid':
                   diam_avg = np.mean(diams, axis=1)  # Average diameter for an ellipsoid.
                   vol = np.pi * diam_avg[0]*diam_avg[1]*diam_avg[2] / 6.0  # Volume of an ellipsoid.
               else:
                   print("Geometry not considered.")
               # Append count based on the target volume and ratio.
               count.append(int(targetvol * ratio / vol))
       else:
           # Default handling for a single geometry type.
           idmin = (np.abs(cfd - 0.5)).argmin()
           if geometry == 'sphere':
               vol = np.pi * diams[idmin] ** 3 / 6.0
           elif geometry == 'ellipsoid':              
               vol = np.pi * diams[0][idmin]*diams[1][idmin]*diams[2][idmin] / 6.0
           else:
               print("Geometry not considered.")
           count.append(int(targetvol / vol))

       return count

    def _make_checkpoints(self, count: List[int]):
        """
        Generate checkpoints for tracking the progress of sphere generation.

        Parameters:
        - count: List of sphere counts for different sizes.

        Returns:
        - An integer representing the checkpoint interval.
        """
        # Calculate the highest count and determine its number of digits.
        maximum = max(count)
        digits = len(str(maximum))

        # Generate a checkpoint value, e.g., 1000 for a maximum of 900.
        return int('1' + '0'*digits)


    def add_spheres_gravity_cfd_individual(self, target_porosity: float,
                                           diams: npt.NDArray[np.float64],
                                           cfd: npt.NDArray[np.float64],
                                           ratios: List[float] = None,
                                           overlap: float = 0.0,
                                           mode: int = 0):
        """
        Add individual spheres to the structure using a gravity packing method.

        Parameters:
        - target_porosity: Desired porosity of the structure.
        - diams: Array of possible diameters.
        - cfd: Cumulative frequency distribution of diameters.
        - ratios: Ratios for different sizes (if specified).
        - overlap: Degree of allowed overlap between spheres.
        - mode: Specific mode for size selection.
        """
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')

        counts = self._estimate_count(target_porosity, diams, cfd, ratios)
        checkpoint = self._make_checkpoints(counts)
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                cumsum = np.cumsum(ratios)
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            cumsum = [1.0]
            self.statistics['diameters'][str(1)] = []
        

        # Check if the structure is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')

        # Estimate counts for each diameter group and define a checkpoint.
        counts = self._estimate_count(target_porosity, diams, cfd, ratios)
        checkpoint = self._make_checkpoints(counts)

        # Validate ratios if provided.
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                # Create cumulative sum for range-based mode selection.
                cumsum = np.cumsum(ratios)
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            cumsum = [1.0]  # Default cumulative sum for a single group.
            self.statistics['diameters'][str(1)] = []

        n_balls = 0  # Counter for the number of balls generated.
        counter = [0 for _ in range(len(counts))]  # Track counts for each group.

        # Main loop to add spheres until the desired porosity is achieved.
        while target_porosity < self.statistics['porosity']:
            # Randomly choose a mode based on the cumulative sum.
            if mode:
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]

            voxel_val = mode * checkpoint  # Set voxel value based on mode and checkpoint.
            rnd2 = np.random.random_sample()  # Randomly sample for diameter selection.
            idmin = (np.abs(cfd[mode] - rnd2)).argmin()
            diam = diams[idmin]
            rad = int(diam * 0.5 / self.voxel + 0.5)  # Calculate radius in voxels.
            clearance = -1 * int(diam * overlap + 0.5)  # Clearance value for overlapping.

            poros1 = self.calculate_porosity(self.im)  # Calculate initial porosity.
            try:
                # Generate a new sphere using pseudo-gravity packing.
                newball = (voxel_val + counter[mode]) * ps.generators.pseudo_gravity_packing(
                    self.im == 0, rad, clearance=clearance, axis=0, maxiter=1, edges='extended')

                # Add the sphere to the structure.
                self.im[np.nonzero(newball)] = 0
                self.im += newball.astype(np.uint16)
            except ValueError:
                # Continue if a ValueError is raised during sphere generation.
                continue

            poros2 = self.calculate_porosity(self.im)  # Calculate updated porosity.

            if poros1 > poros2:  # If porosity has decreased, update statistics.
                self.statistics['porosity'] = poros2
                self.statistics['diameters'][str(mode + 1)].append(diam)
                self.statistics['diameters']['Total'].append(diam)
                counter[mode] += 1  # Increment counter for the current mode.

            # Increment the sphere counter and print status.
            n_balls += 1
            print("Balls created: ", n_balls)
            print(f"Porosity: {self.statistics['porosity']:.2f}")
            print(f"Diameter of the ball: {self.statistics['diameters']['Total'][-1]}")
            print("")
    
    def add_spheres_gravity_cfd_counted(self, target_porosity: float,
                                        diams: npt.NDArray[np.float64],
                                        cfd: npt.NDArray[np.float64],
                                        ratios: List[float] = None,
                                        overlap: float = 0.0):

        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')

        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            self.statistics['diameters'][str(1)] = []

        particle_diameters = self._estimate_number(target_porosity, diams, cfd, ratios, particle='sphere')
        while any(sublist for sublist in particle_diameters):
            rnd = np.random.random_sample()

            if rnd < 0.5 and particle_diameters[0] or ratios is None:
                diam = particle_diameters[0].pop(0)
                mode = 0
            elif rnd > 0.5 and particle_diameters[1] and ratios is not None:
                diam = particle_diameters[1].pop(0)
                mode = 1
            else:
                continue

            rad = int(diam * 0.5 / self.voxel + 0.5)
            clearance = -1 * int(diam * overlap + 0.5)

            poros1 = self.calculate_porosity(self.im)
            try:
                newball = (mode + 1) * ps.generators.pseudo_gravity_packing(self.im == 0, rad, clearance=clearance,
                                                                            axis=0, maxiter=1, edges='extended')
                
                center = calc_centroid(newball)
                self.im[np.nonzero(newball)] = 0
                self.im += newball.astype(np.uint16)
            except ValueError:
                continue

            poros2 = self.calculate_porosity(self.im)

            if poros1 > poros2:
                self.statistics['porosity'] = poros2
                self.statistics['diameters'][str(mode + 1)].append(diam)
                self.statistics['diameters']['Total'].append(diam)
                self.data['id'].append(mode + 1)
                self.data['x'].append(center[0])
                self.data['y'].append(center[1])
                self.data['z'].append(center[2])
                self.data['r'].append(0.5 * diam)

    def add_spheres_gravity_cfd(self, target_porosity: float,
                                diams: npt.NDArray[np.float64],
                                cfd: npt.NDArray[np.float64],
                                ratios: List[float] = None,
                                overlap: float = 0.0, 
                                max_tries: int = 100,
                                axis: int = 0,
                                mode: int = 0,
                                counts_break: bool = False):
        """
        Adds spheres to the structure using a gravity-based packing method until the desired target porosity is reached.
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible sphere diameters.
        - cfd: Cumulative frequency distribution for the sphere diameters.
        - ratios: List of ratios specifying the proportion of different sphere sizes.
        - overlap: Allowed overlap between spheres (0.0 means no overlap).
        - max_tries: Maximum number of attempts to place a sphere before stopping.
        - axis: Axis along which spheres are aligned (default is 0); 0(x), 1(y) 2(z).
        - mode: If set, enables random size selection based on ratios (used for 2 types of AM).
        - counts_break: If set to True, stops once a specified number of spheres is reached.
        """
        # Check if the structure's image is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Handle cases where ratios are provided to determine different sphere size groups.
        if ratios is not None:
            # Ensure that the sum of ratios is equal to 1.0.
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                # Create a cumulative sum of ratios for probabilistic selection.
                cumsum = np.cumsum(ratios)
                # Initialize lists to store diameters for each ratio group.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            # If no ratios are provided, assume a single group.
            cumsum = [1.0]
            self.statistics['diameters'][str(1)] = []
    
        # Estimate the number of spheres required for each size group.
        counts = self._estimate_count(target_porosity, diams, cfd, ratios)
        n_balls = 0  # Counter to track the total number of spheres added.
        tries = 0  # Counter for tracking consecutive failed placement attempts.
    
        # Add spheres until the target porosity is reached.
        while target_porosity < self.statistics['porosity']:
            start_time = time.time()  # Track the start time for timeout control.
    
            # Select a mode (size group) based on ratios if specified.
            if mode:
                rnd1 = np.random.random_sample()  # Randomly select a mode.
                # Determine the mode based on cumulative sum values.
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
          
            # Randomly select a sphere diameter based on the cumulative frequency distribution.
            rnd2 = np.random.random_sample()
            idmin = (np.abs(cfd[mode] - rnd2)).argmin()  # Find the nearest diameter index.
            diam = diams[idmin]  # Select the corresponding diameter.
            rad = int(diam * 0.5 / self.voxel + 0.5)  # Calculate the radius in voxels.
            clearance = -1 * int(diam * overlap + 0.5)  # Set the clearance based on overlap.
    
            poros1 = self.calculate_porosity(self.im)  # Calculate the current porosity.
    
            try:
                # Generate a new sphere using pseudo-gravity packing.
                newball = (mode + 1) * ps.generators.pseudo_gravity_packing(self.im == 0, rad, clearance=clearance,
                                                                            axis=axis, maxiter=1, edges='extended')
                # Update the structure's image with the new sphere.
                self.im[np.nonzero(newball)] = 0  # Reset previous sphere positions to avoid overlap.
                self.im += newball.astype(np.uint16)  # Add the new sphere to the structure.
            except ValueError:
                # Handle cases where a sphere could not be placed successfully.
                tries += 1  # Increment the failure counter.
                if tries >= max_tries:  # Stop if the maximum number of attempts is reached.
                    print("Value error. Nº of tries: ", tries)
                    break
                if time.time() - start_time > 10:  # Stop if the time limit is exceeded.
                    print("Time exceeded.")
                    break
                continue  # Skip to the next iteration if placement fails.
    
            poros2 = self.calculate_porosity(self.im)  # Calculate the new porosity after placement.
    
            # If the new porosity is lower (more filled space), update statistics.
            if poros1 > poros2:
                self.statistics['porosity'] = poros2  # Update the porosity value.
                # Store the diameter of the newly added sphere.
                self.statistics['diameters'][str(mode + 1)].append(diam)
                self.statistics['diameters']['Total'].append(diam)
    
            # Increment the total number of spheres created.
            n_balls += 1
            # print("Tries: ", tries)
            # print("Time:", time.time()-start_time)
            tries = 0 # Reset the failure counter after a successful placement.
            # print("Balls created: ", n_balls)
            # porosity_ratio = target_porosity / self.statistics['porosity']
            # print(f"Porosity: {self.statistics['porosity']:.5f}")
            # print(f"Diameter of the ball: {self.statistics['diameters']['Total'][-1]:.2f} um")
            # print("")
            # If `counts_break` is set to True, stop if the number of spheres exceeds a threshold.
            if counts_break:
                if n_balls >= counts[0] + 80:  # Stop if the threshold is exceeded.
                    break
            
    def add_spheres_rsa_cfd(self, target_porosity: float, 
                            diams: npt.NDArray[np.float64],
                            cfd: npt.NDArray[np.float64], 
                            ratios: List[float] = None, 
                            overlap: float = 0.0,
                            electrode_generation_mode: str = 'extended',
                            mode: int = 0):
        """
        Adds spheres to the structure using the Random Sequential Addition (RSA) method until the desired target porosity is reached.
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible sphere diameters.
        - cfd: Cumulative frequency distribution for the sphere diameters.
        - ratios: List of ratios specifying the proportion of different sphere sizes.
        - overlap: Allowed overlap between spheres (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - mode: If set, enables random size selection based on ratios (used for 2 types of AM).
        """
        # Check if the structure's image is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Handle cases where ratios are provided to determine different sphere size groups.
        if ratios is not None:
            # Ensure that the sum of ratios is equal to 1.0.
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                # Create a cumulative sum of ratios for probabilistic selection.
                cumsum = np.cumsum(ratios)
                # Initialize lists to store diameters for each ratio group.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            # If no ratios are provided, assume a single group.
            cumsum = [1.0]
            self.statistics['diameters'][str(1)] = []
    
        n_balls = 0  # Counter to track the total number of spheres added.
    
        # Add spheres until the target porosity is reached.
        while target_porosity < self.statistics['porosity']:
            # Select a mode (size group) based on ratios if specified.
            if mode:
                rnd1 = np.random.random_sample()  # Randomly select a mode.
                # Determine the mode based on cumulative sum values.
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
    
            # Randomly select a sphere diameter based on the cumulative frequency distribution.
            rnd2 = np.random.random_sample()
            idmin = (np.abs(cfd[mode] - rnd2)).argmin()  # Find the nearest diameter index.
            diam = diams[idmin]  # Select the corresponding diameter.
            rad = int(diam * 0.5 / self.voxel + 0.5)  # Calculate the radius in voxels.
            protrusion = int(diam * overlap + 0.5)  # Calculate the allowed protrusion based on overlap.
    
            poros1 = self.calculate_porosity(self.im)  # Calculate the current porosity.
    
            # Generate a new sphere using the RSA (Random Sequential Addition) method.
            self.im = (mode + 1) * ps.generators.rsa(im_or_shape=self.im, r=rad, n_max=1,
                                                     mode=electrode_generation_mode,
                                                     protrusion=protrusion)
            
            poros2 = self.calculate_porosity(self.im)  # Calculate the new porosity after placement.
    
            # If the new porosity is lower (more filled space), update statistics.
            if poros1 > poros2:
                self.statistics['porosity'] = poros2  # Update the porosity value.
                # Store the diameter of the newly added sphere.
                self.statistics['diameters'][str(mode + 1)].append(diam)
                self.statistics['diameters']['Total'].append(diam)
            else:
                # If porosity didn't change, continue to the next iteration.
                continue
    
            # Increment the total number of spheres created.
            n_balls += 1
    
            # Print statistics every 10 spheres.
            if n_balls % 10 == 0:
                print(" Balls created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Diameter of the ball: {self.statistics['diameters']['Total'][-1]:.2f} um")
                print("")
            # if n_balls >= counts[0] + 80:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            # if n_balls >= 30:
                # break
    
        # Ensure that all sphere values are set to 1 in the final image.
        self.im[self.im >= 1] = 1
        self.im.astype(np.uint16)  # Convert image to 16-bit unsigned integer type.

    def add_spheres_rsa_cfd_individual(self, target_porosity: float, 
                                       diams: npt.NDArray[np.float64],
                                       cfd: npt.NDArray[np.float64], 
                                       ratios: List[float] = None, 
                                       overlap: float = 0.0,
                                       electrode_generation_mode: str = 'extended',
                                       angles: int = 0,
                                       mode: int = 0):
        """
        Adds individual spheres to the structure using the Random Sequential Addition (RSA) method until the desired target porosity is reached.
        Each sphere is uniquely labeled, allowing for individual tracking within the structure.
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible sphere diameters.
        - cfd: Cumulative frequency distribution for sphere diameters.
        - ratios: List of ratios specifying the proportion of different sphere sizes.
        - overlap: Allowed overlap between spheres (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angles: If set, records random orientation angles for each sphere.
        - mode: If set, enables random size selection based on ratios.
        """
        # Check if the structure's image (`self.im`) is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Handle cases where ratios are provided to determine different sphere size groups.
        if ratios is not None:
            # Ensure that the sum of ratios is equal to 1.0.
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                # Create a cumulative sum of ratios for probabilistic selection.
                cumsum = np.cumsum(ratios)
                # Initialize lists to store diameters for each ratio group.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            # If no ratios are provided, assume a single group.
            cumsum = [1.0]
            self.statistics['diameters'][str(1)] = []
    
        # Initialize a list to store sphere angles if the `angles` parameter is set.
        if angles:
            self.statistics['angles'] = []
    
        n_balls = 0  # Counter to track the total number of spheres added.
    
        # Add spheres until the target porosity is reached.
        while target_porosity < self.statistics['porosity']:
            # Select a mode (size group) based on ratios if specified.
            if mode:
                rnd1 = np.random.random_sample()  # Generate a random number to select the mode.
                # Determine the mode based on cumulative sum values.
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
    
            # Randomly select a sphere diameter based on the cumulative frequency distribution.
            rnd2 = np.random.random_sample()
            idmin = (np.abs(cfd[mode] - rnd2)).argmin()  # Find the nearest diameter index.
            diam = diams[idmin]  # Select the corresponding diameter.
            rad = int(diam * 0.5 / self.voxel + 0.5)  # Calculate the radius in voxels.
            protrusion = int(diam * overlap + 0.5)  # Calculate the allowed protrusion based on overlap.
    
            poros1 = self.calculate_porosity(self.im)  # Calculate the current porosity.
    
            # Determine the label for the new sphere. Each sphere is uniquely labeled.
            index = np.max(self.im) + 1  # Find the next available label.
            
            # If the calculated index jumps too far ahead, correct it.
            if index > n_balls + 2:
                # Adjust labels in the image to maintain consistent indexing.
                self.im[self.im > n_balls] = n_balls
                index = n_balls + 2
    
            # Ensure that the first sphere has a correct initial label.
            if n_balls == 0 and index == 1:
                index = 2
    
            # Try to generate a new sphere using RSA. If generation fails, skip to the next iteration.
            try:
                ball = ps.generators.rsa(im_or_shape=self.im, r=rad, n_max=1,
                                         mode=electrode_generation_mode,
                                         protrusion=protrusion)
            except:
                continue
    
            # Insert the new sphere into the image and label it with the unique index.
            self.im[np.where(ball == 1)] = index
            poros2 = self.calculate_porosity(self.im)  # Calculate the new porosity after placement.
    
            # If the new porosity is lower (more filled space), update statistics.
            if poros1 > poros2:
                self.statistics['porosity'] = poros2  # Update the porosity value.
                # Store the diameter of the newly added sphere.
                self.statistics['diameters'][str(mode + 1)].append(diam)
                self.statistics['diameters']['Total'].append(diam)
    
                # Record a random orientation angle if the `angles` parameter is set.
                if angles:
                    self.statistics['angles'].append(random.randint(0, 2 * np.pi))
            else:
                # If porosity didn't change, continue to the next iteration.
                continue
    
            """ A counter of the number of balls generated """
            n_balls += 1
    
            # Print statistics every 100 spheres.
            if n_balls % 100 == 0:
                print(" Balls created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Diameter of the ball: {self.statistics['diameters']['Total'][-1]:.2f} um")
                print("")
    
        # Final adjustment: decrement labels by 1 to ensure consistency.
        self.im[self.im > 1] -= 1
        self.im.astype(np.uint16)  # Convert image to 16-bit unsigned integer type.

    def add_spheres_rsa(self, target_porosity: float, 
                        psd_type: str, 
                        max_diam: float,
                        x0: float, 
                        sigma: float, 
                        overlap: float = 0.0):
        """
        Adds spheres to the structure using the Random Sequential Addition (RSA) method until the desired target porosity is reached.
        Sphere diameters are determined based on the specified Probability Density Function (PDF) type, which can be either 'gauss' or 'lognormal'.
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - psd_type: Type of Probability Density Function (PDF) to use ('gauss' or 'lognormal').
        - max_diam: Maximum diameter for the spheres.
        - x0: Mean diameter for the chosen PDF.
        - sigma: Standard deviation for the chosen PDF.
        - overlap: Allowed overlap between spheres (0.0 means no overlap).
        """
        # Check if the structure's image (`self.im`) is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Determine the Probability Density Function (PDF) type based on `psd_type`.
        if psd_type == 'gauss':
            psd = gauss  # Use Gaussian distribution for sphere diameters.
        elif psd_type == 'lognormal':
            psd = lognormal  # Use Lognormal distribution for sphere diameters.
        else:
            raise ValueError('Insert "gauss" or "lognormal" as psd_type')
    
        n_balls = 0  # Counter to track the total number of spheres added.
    
        # Add spheres until the target porosity is achieved.
        while target_porosity < self.statistics['porosity']:
            random_number = np.random.random_sample()  # Generate a random number for probability check.
            random_diameter = np.random.random_sample() * max_diam  # Randomly select a diameter up to `max_diam`.
            random_radius_voxel = int(random_diameter * 0.5 / self.voxel + 0.5)  # Convert diameter to radius in voxels.
    
            # Skip spheres that are too small (radius equals zero in voxel space).
            if random_radius_voxel == 0:
                continue
    
            """ Calculate clearance value for overlap handling """
            clearance_value = int(random_radius_voxel * overlap + 0.5)
    
            # Calculate the current porosity of the structure.
            poros1 = self.calculate_porosity(self.im)
    
            # Check if the random diameter falls within the specified PDF.
            if random_number < psd(x=random_diameter, x0=x0, sigma=sigma):
                # Generate and add a new sphere using the RSA method.
                self.im = ps.generators.rsa(im_or_shape=self.im, r=random_radius_voxel, n_max=1,
                                            mode='extended', clearance=clearance_value)
    
            # Calculate the new porosity after adding the sphere.
            poros2 = self.calculate_porosity(self.im)
            # Update the porosity if the new sphere decreases it (i.e., more filled volume).
            if poros1 > poros2:
                self.statistics['porosity'] = poros2
    
            n_balls += 1  # Increment the sphere counter.
    
        # Convert the image to binary format (0 or 1) and cast to 16-bit unsigned integers.
        self.im = 1 * self.im
        self.im.astype(np.uint16)

    
    def add_ellipsoids_rsa_cfd(self, target_porosity: float, 
                               diams: npt.NDArray[np.float64],
                               cfd: npt.NDArray[np.float64], 
                               ratios: List[float] = None, 
                               overlap: float = 0.0,
                               electrode_generation_mode: str = 'extended',
                               angle_tolerance_flag: bool = True,
                               angle_tolerance: int = 5, 
                               theta_x: int = 0,
                               theta_y: int = 0,
                               theta_z: int = 0,
                               mode: int = 0):
        """
        Adds ellipsoids to the structure using Random Sequential Addition (RSA) method based on the Cumulative Frequency Distribution (CFD).
        Allows for control over the orientation of ellipsoids through random angles or fixed values.
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible ellipsoid diameters for the three axes.
        - cfd: Cumulative Frequency Distribution for each axis.
        - ratios: Proportions for different groups of ellipsoids (must sum to 1.0).
        - overlap: Allowed overlap between ellipsoids (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angle_tolerance_flag: Whether to enable random orientation within a specified range.
        - angle_tolerance: Range for random orientation if `angle_tolerance_flag` is enabled.
        - theta_x, theta_y, theta_z: Fixed orientation angles if `angle_tolerance_flag` is disabled.
        - mode: If set, enables random size selection based on ratios.
        """
        # Check if the structure's image (`self.im`) is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Handle cases where ratios are provided to define ellipsoid size groups.
        if ratios is not None:
            # Ensure that the sum of ratios is equal to 1.0.
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                # Create a cumulative sum for probabilistic selection of ellipsoid sizes.
                cumsum = np.cumsum(ratios)
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []
        else:
            # Default behavior for a single group.
            cumsum = [1.0]
            self.statistics['diameters']['x'] = []
            self.statistics['diameters']['y'] = []
            self.statistics['diameters']['z'] = []
            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
    
        n_balls = 0  # Counter for the number of ellipsoids added.
        ell_not_cre_counter = 0  # Counter to track failed ellipsoid creation attempts.
    
        # Add ellipsoids until the target porosity is achieved.
        while target_porosity < self.statistics['porosity']:
            if mode:
                # Select a size group probabilistically based on `ratios`.
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
    
            # Randomly select diameters for the three axes.
            diams_list = []
            radii = []
            for i in range(3):
                rnd2 = np.random.random_sample()
                idmin = (np.abs(cfd[mode] - rnd2)).argmin()  # Select diameter index based on CFD.
                diams_list.append(diams[i][idmin])  # Store the selected diameter.
                radii.append(int(diams_list[i] * 0.5 / self.voxel + 0.5))  # Convert to radius in voxels.
    
            poros1 = self.calculate_porosity(self.im)  # Calculate the current porosity.
    
            # Generate random angles for orientation if `angle_tolerance_flag` is enabled.
            if angle_tolerance_flag:
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)
                theta_x = random.choice(chosen_range_x)
                theta_y = random.choice(chosen_range_y)
            else:
                # Use fixed angles if random orientation is disabled.
                theta_x = theta_x
                theta_y = theta_y
            theta_z = 0
    
            protrusion = int(min(radii) * 2 * overlap + 0.5)  # Calculate overlap protrusion.
    
            try:
                # Generate and place a new ellipsoid with the specified radii and angles.
                ellipsoid, angles = ps.generators.rsa_ellipsoids(im_or_shape=self.im, rA=radii[0], rB=radii[1], rC=radii[2], 
                                                                n_max=1, mode=electrode_generation_mode, protrusion=protrusion, 
                                                                smooth=True, rotation_angles=True, theta_x=np.radians(theta_x), 
                                                                theta_y=np.radians(theta_y), theta_z=np.radians(theta_z))
                self.im = ellipsoid * (mode + 1)
            except:
                # If ellipsoid generation fails, increment the failure counter.
                ell_not_cre_counter += 1
                if ell_not_cre_counter > 100:  # Stop if too many consecutive failures occur.
                    print("Structure generation stopped.")
                    break
                continue
    
            # Check if the new ellipsoid reduces porosity.
            poros2 = self.calculate_porosity(self.im)
            if poros1 > poros2:
                ell_not_cre_counter = 0  # Reset failure counter on successful addition.
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['x'].append(diams_list[0])
                self.statistics['diameters']['y'].append(diams_list[1])
                self.statistics['diameters']['z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                self.statistics['angles']['theta_x'].append(angles[0])
                self.statistics['angles']['theta_y'].append(angles[1])
                self.statistics['angles']['theta_z'].append(angles[2])
            else:
                continue
    
            n_balls += 1  # Increment the ellipsoid counter.
            if n_balls % 100 == 0:
                print("Ellipsoids created: ", n_balls)
                print(f"Porosity: {self.statistics['porosity']:.4f}")
                print(f"Ellipsoid of size: {radii[0]}, {radii[1]}, {radii[2]} inserted.")
                print("")
    
        # Convert the image to binary format (0 or 1) and cast to 16-bit unsigned integers.
        self.im[self.im >= 1] = 1
        self.im.astype(np.uint16)


    def add_ellipsoids_rsa_cfd_individual(self, target_porosity: float, 
                                           diams: npt.NDArray[np.float64],
                                           cfd: npt.NDArray[np.float64], 
                                           ratios: List[float] = None, 
                                           overlap: float = 0.0,
                                           electrode_generation_mode: str = 'extended',
                                           angle_tolerance_flag: bool = True,
                                           angle_tolerance: int = 5, 
                                           angles_range_flag: bool = True,
                                           theta_x: int = 0,
                                           theta_y: int = 0,
                                           theta_z: int = 0,
                                           mode: int = 0):
        """
        Generates and adds ellipsoids to the structure while keeping track of porosity.
        Each generated ellipsoid is assigned a unique label number.
        
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible ellipsoid diameters for the three axes.
        - cfd: Cumulative Frequency Distribution for each axis.
        - ratios: Proportions for different groups of ellipsoids (must sum to 1.0).
        - overlap: Allowed overlap between ellipsoids (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angle_tolerance_flag: Whether to enable random orientation within a specified range.
        - angle_tolerance: Range for random orientation if `angle_tolerance_flag` is enabled.
        - theta_x, theta_y, theta_z: Fixed orientation angles if `angle_tolerance_flag` is disabled.
        - mode: If set, enables random size selection based on ratios.
        """
        
        # Check if the image (structure) has been defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Validate the ratios, if provided.
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')  # Ensure ratios sum to 1.
            else:
                cumsum = np.cumsum(ratios)  # Compute cumulative sum of ratios.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []  # Initialize diameter lists.
        else:
            # If no ratios are provided, initialize empty lists for diameter and angle statistics.
            cumsum = [1.0]
            self.statistics['diameters']['x'] = []
            self.statistics['diameters']['y'] = []
            self.statistics['diameters']['z'] = []
            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
    
        n_balls = 0  # Counter for generated ellipsoids.
        ell_not_cre_counter = 0  # Counter for unsuccessful ellipsoid creations.
        vol_ratio = 1.00  # Initial volume ratio threshold.
        max_vol_fact_c = 0  # Tracks the maximum volume factor counter.
        vol_fact_c = 0
        print(f" Volume ratio: {vol_ratio}")
    
        # Loop until the target porosity is reached.
        while target_porosity < self.statistics['porosity']:
            if mode:
                # Randomly select a mode based on cumulative sums.
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
            
            # Calculate the current porosity of the structure.
            poros1 = self.calculate_porosity(self.im)
    
            # Initialize lists for diameters and radii.
            diams_list = [[], [], []]
            radii = []
            radii_aux = []
            
            # Generate random factors for ellipsoid dimensions.
            factor_y = random.uniform(0.5, 0.75)
            factor_z = random.uniform(0.15, 0.4)
            diams_sph = random.uniform(6.5, 14.5)  # Random sphere diameter.
            vols = 4 * np.pi * (diams_sph / 2)**3 / 3  # Volume of the sphere.
            diams_x = 2 * (3 * vols / (4 * np.pi * factor_y * factor_z))**(1/3)  # Calculate x-diameter.
            diams_y = factor_y * diams_x  # Calculate y-diameter based on factor.
            diams_z = factor_z * diams_x  # Calculate z-diameter based on factor.                
            # Calculate radii in voxel units.
            radii = [diams_x * 0.5 / self.voxel,
                      diams_y * 0.5 / self.voxel,
                      diams_z * 0.5 / self.voxel]
            diams_list[0] = diams_x
            diams_list[1] = diams_y
            diams_list[2] = diams_z
    
            # Determine the index of the diameter to sample based on current porosity.
            if poros1 > 2 * target_porosity:
                idmin = random.randint(int(len(diams[0]) * 0.6), len(diams[0]) - 1)
            elif poros1 > 1.5 * target_porosity and poros1 < 2 * target_porosity:
                idmin = random.randint(int(len(diams[0]) * 0.1), int(len(diams[0]) * 0.5))
            else:
                idmin = random.randint(int(len(diams[0]) * 0.075), len(diams[0]) - 1)
    
            # Retrieve diameters and compute radii.
            # rnd2 = np.random.random_sample()
            for i in range(3):
                # idmin = (np.abs(cfd[mode] - rnd2)).argmin()
                diams_list.append(diams[i][idmin])  # Append diameter to the list.
                radii_aux.append(round(diams_list[i] * 0.5 / self.voxel + 0.5))  # Adjust radius.
                radii.append(diams_list[i] * 0.5 / self.voxel)  # Append radius in voxel units.
    
            el_vol = 4 / 3 * np.pi * radii[0] * radii[1] * radii[2]  # Calculate ellipsoid volume.
    
            # Check angle tolerances and select random angles for rotation if enabled.
            if angle_tolerance_flag:
                angle_tolerance = angle_tolerance
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                theta_x = random.choice(chosen_range_x)  # Select random x-angle.
                theta_y = random.choice(chosen_range_y)  # Select random y-angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                
                index = np.max(self.im) + 1  # Assign a new index for the ellipsoid.
                if index > n_balls + 2:
                    self.im[self.im > n_balls] = n_balls  # Reassign existing ellipsoids to avoid conflicts.
                    index = n_balls + 1  # Increment index for the current ellipsoid.
                if n_balls == 0 and index == 1:
                    index = 2  # Ensure unique index for the first ellipsoid.
                    
            # Check if angle ranges are specified and select angles accordingly.
            if angles_range_flag and not angle_tolerance_flag:        
                # Define angle ranges for x and y rotations.
                theta_x_values = [-10, -5, 0, 5, 10]
                theta_y_values = [-10, -5, 0, 5, 10]
                combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))  # Generate all combinations.
                combination_indices = {combination: index + 2 for index, combination in enumerate(combinations)}
                
                self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]  # Store x angles.
                self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]  # Store y angles.
                self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]  # Zero z angles.
    
                theta_x = random.choice(theta_x_values)  # Randomly choose x angle.
                theta_y = random.choice(theta_y_values)  # Randomly choose y angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                
                index = combination_indices[(theta_x, theta_y)]  # Assign index based on angle combination.
            else:    
                # If no angle range is specified, use provided angles.
                theta_x = theta_x
                theta_y = theta_y
                theta_z = theta_z
                index = n_balls + 2  # Assign index for the ellipsoid.
    
            # Determine protrusion based on maximum radius and overlap.
            protrusion = int(max(radii) * 2 * overlap + 0.5)
    
            # Attempt to generate the ellipsoid with specified parameters.
            try:
                ellipsoid, angles = ps.generators.rsa_ellipsoids(im_or_shape=self.im, rA=round(radii[0]), rB=round(radii[1]), rC=round(radii[2]), 
                                                                 n_max=1, mode=electrode_generation_mode, protrusion=protrusion, 
                                                                 smooth=True, rotation_angles=True, theta_x=np.radians(theta_x), 
                                                                 theta_y=np.radians(theta_y), theta_z=np.radians(theta_z))    
            except:
                # Increment failure counter if ellipsoid generation fails.
                ell_not_cre_counter += 1
                print(' Warning: not creating an ellipsoid')
                continue  # Skip to next iteration.
    
            # Check front and back faces (z=0 and z=max)
            front_face = ellipsoid[:, :, 0] == index
            back_face = ellipsoid[:, :, -1] == index
            
            # Check left and right faces (y=0 and y=max)
            left_face = ellipsoid[:, 0, :] == index
            right_face = ellipsoid[:, -1, :] == index
            
            # Check top and bottom faces (x=0 and x=max)
            top_face = ellipsoid[0, :, :] == index
            bottom_face = ellipsoid[-1, :, :] == index
            
            # Combine all boundary checks
            boundary_check = np.any(front_face) or np.any(back_face) or np.any(left_face) or np.any(right_face) or np.any(top_face) or np.any(bottom_face)
            print(f" Boundary has {index}:", boundary_check)  # Output will be True or False
    
            if (len(np.where(ellipsoid==1)[0]) < vol_ratio*el_vol) and not boundary_check:
                index = 0
                vol_fact_c +=1
                if vol_fact_c >= 100:
                    vol_ratio -= 0.025
                    print(f" Volume factor overcome. {vol_ratio:.2f}, {vol_fact_c}")
                    print(" -------------------------------------------------------------------------")
            else:
                print(" Ellipsoid kept")
                print(f" Particle id: {index}")
                print(f" Original volume: {el_vol:.2f}")
                print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
                print(f" Volume factor: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
                print(f" Porosity: {poros1*100:.2f} %")
                print(f" Volume ratio: {vol_ratio:.2f}")
                print(f" Volume factor counter: {vol_fact_c}")
                if vol_fact_c > max_vol_fact_c:
                    max_vol_fact_c = vol_fact_c
                vol_fact_c = 0
                print(f" Maximum volume factor counter: {max_vol_fact_c}")
                print(f" Radio a: {radii[0]:.2f}; b: {radii[1]:.2f}; c: {radii[2]:.2f}")
                print(" -------------------------------------------------------------------------")    
            
            self.im[np.where(ellipsoid == 1)] = index
            poros2 = self.calculate_porosity(self.im)
            if poros1 > poros2:
                ell_not_cre_counter = 0
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['x'].append(diams_list[0])
                self.statistics['diameters']['y'].append(diams_list[1])
                self.statistics['diameters']['z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                if angle_tolerance_flag == True and angles_range_flag == False:
                    self.statistics['angles']['theta_x'].append(angles[0])
                    self.statistics['angles']['theta_y'].append(angles[1])
                    self.statistics['angles']['theta_z'].append(angles[2])
                """ A counter of the number of balls generated"""
                n_balls += 1
                
            # if n_balls % 100 == 0:
            #     print(" Ellipsoids created: ", n_balls)
            #     print(f" Porosity: {self.statistics['porosity']:.4f}")
            #     print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
            #     print("")
            # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
            # if n_balls >= 10:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            #     break     

        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)  
    
    def add_ellipsoids_rsa_cfd_counted_individual(self, target_porosity: float, 
                                                  diams: npt.NDArray[np.float64],
                                                  cfd: npt.NDArray[np.float64], 
                                                  ratios: List[float] = None, 
                                                  overlap: float = 0.0,
                                                  electrode_generation_mode: str = 'extended',
                                                  angle_tolerance_flag: bool = True,
                                                  angle_tolerance: int = 5, 
                                                  angles_range_flag: bool = True,
                                                  theta_x: int = 0,
                                                  theta_y: int = 0,
                                                  theta_z: int = 0,
                                                  mode: int = 0):
        """
        Generates and adds ellipsoids to the structure while keeping track of porosity.
        Each generated ellipsoid is assigned a unique label number.
        
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible ellipsoid diameters for the three axes.
        - cfd: Cumulative Frequency Distribution for each axis.
        - ratios: Proportions for different groups of ellipsoids (must sum to 1.0).
        - overlap: Allowed overlap between ellipsoids (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angle_tolerance_flag: Whether to enable random orientation within a specified range.
        - angle_tolerance: Range for random orientation if `angle_tolerance_flag` is enabled.
        - theta_x, theta_y, theta_z: Fixed orientation angles if `angle_tolerance_flag` is disabled.
        - mode: If set, enables random size selection based on ratios.
        """
       
        # Check if the image (structure) has been defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
        
        # Validate the ratios, if provided.
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')  # Ensure ratios sum to 1.
            else:
                cumsum = np.cumsum(ratios)  # Compute cumulative sum of ratios.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []  # Initialize diameter lists.
        else:
            # If no ratios are provided, initialize empty lists for diameter and angle statistics.
            cumsum = [1.0]
            self.statistics['diameters']['x'] = []
            self.statistics['diameters']['y'] = []
            self.statistics['diameters']['z'] = []
            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
        
        n_balls = 0  # Counter for generated ellipsoids.
        ell_not_cre_counter = 0  # Counter for unsuccessful ellipsoid creations.
        vol_ratio = 1.00  # Initial volume ratio threshold.
        max_vol_fact_c = 0  # Tracks the maximum volume factor counter.
        vol_fact_c = 0
        print(f" Volume ratio: {vol_ratio}")
        
        # Loop until the target porosity is reached.
        particle_diameters = self._estimate_number(target_porosity, diams, cfd, 
                                                    ratios, particle='ellipsoid')
        # while any(sublist for sublist in particle_diameters) or target_porosity < self.statistics['porosity']:
        while target_porosity < self.statistics['porosity']:
            if particle_diameters[0]:
                diams_x, diams_y, diams_z = particle_diameters[0].pop(0)
            else:
                print(" Warning: No more elements to pop from particle_diameters!")
                break 
            
            # Calculate the current porosity of the structure.
            poros1 = self.calculate_porosity(self.im)
        
            # Initialize lists for diameters and radii.
            diams_list = [[], [], []]
            radii = []
            
            # Calculate radii in voxel units.
            radii = [diams_x * 0.5 / self.voxel,
                      diams_y * 0.5 / self.voxel,
                      diams_z * 0.5 / self.voxel]
            diams_list[0] = diams_x
            diams_list[1] = diams_y
            diams_list[2] = diams_z
            el_vol = 4 / 3 * np.pi * radii[0] * radii[1] * radii[2]  # Calculate ellipsoid volume.

            # Check angle tolerances and select random angles for rotation if enabled.
            if angle_tolerance_flag:
                angle_tolerance = angle_tolerance
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                theta_x = random.choice(chosen_range_x)  # Select random x-angle.
                theta_y = random.choice(chosen_range_y)  # Select random y-angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                
                index = np.max(self.im) + 1  # Assign a new index for the ellipsoid.
                if index > n_balls + 2:
                    self.im[self.im > n_balls] = n_balls  # Reassign existing ellipsoids to avoid conflicts.
                    index = n_balls + 1  # Increment index for the current ellipsoid.
                if n_balls == 0 and index == 1:
                    index = 2  # Ensure unique index for the first ellipsoid.
                    
            # Check if angle ranges are specified and select angles accordingly.
            if angles_range_flag and not angle_tolerance_flag:        
                # Define angle ranges for x and y rotations.
                theta_x_values = [-10, -5, 0, 5, 10]
                theta_y_values = [-10, -5, 0, 5, 10]
                combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))  # Generate all combinations.
                combination_indices = {combination: index + 2 for index, combination in enumerate(combinations)}
                
                self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]  # Store x angles.
                self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]  # Store y angles.
                self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]  # Zero z angles.
        
                theta_x = random.choice(theta_x_values)  # Randomly choose x angle.
                theta_y = random.choice(theta_y_values)  # Randomly choose y angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                
                index = combination_indices[(theta_x, theta_y)]  # Assign index based on angle combination.
            else:    
                # If no angle range is specified, use provided angles.
                theta_x = theta_x
                theta_y = theta_y
                theta_z = theta_z
                index = n_balls + 2  # Assign index for the ellipsoid.
        
            # Determine protrusion based on maximum radius and overlap.
            protrusion = int(max(radii) * 2 * overlap + 0.5)
        
            # Attempt to generate the ellipsoid with specified parameters.
            try:
                ellipsoid, angles = ps.generators.rsa_ellipsoids(im_or_shape=self.im, rA=round(radii[0]), rB=round(radii[1]), rC=round(radii[2]), 
                                                                 n_max=1, mode=electrode_generation_mode, protrusion=protrusion, 
                                                                 smooth=True, rotation_angles=True, theta_x=np.radians(theta_x), 
                                                                 theta_y=np.radians(theta_y), theta_z=np.radians(theta_z))    
            except:
                # Increment failure counter if ellipsoid generation fails.
                ell_not_cre_counter += 1
                print(' Warning: not creating an ellipsoid')
                continue  # Skip to next iteration.
        
            # Check front and back faces (z=0 and z=max)
            front_face = ellipsoid[:, :, 0] == index
            back_face = ellipsoid[:, :, -1] == index
            
            # Check left and right faces (y=0 and y=max)
            left_face = ellipsoid[:, 0, :] == index
            right_face = ellipsoid[:, -1, :] == index
            
            # Check top and bottom faces (x=0 and x=max)
            top_face = ellipsoid[0, :, :] == index
            bottom_face = ellipsoid[-1, :, :] == index
            
            # Combine all boundary checks
            boundary_check = np.any(front_face) or np.any(back_face) or np.any(left_face) or np.any(right_face) or np.any(top_face) or np.any(bottom_face)
            print(f" Boundary has {index}:", boundary_check)  # Output will be True or False
        
            if (len(np.where(ellipsoid==1)[0]) < vol_ratio*el_vol) and not boundary_check:
                index = 0
                vol_fact_c +=1
                particle_diameters[0].append([diams_x, diams_y, diams_z])
                if vol_fact_c >= 100:
                    vol_ratio -= 0.025
                    print(f" Volume factor overcome. {vol_ratio:.2f}, {vol_fact_c}")
                    print(" -------------------------------------------------------------------------")
            else:
                print(" Ellipsoid kept")
                print(f" Particle id: {index}")
                print(f" Original volume: {el_vol:.2f}")
                print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
                print(f" Volume factor: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
                print(f" Porosity: {poros1*100:.2f} %")
                print(f" Volume ratio: {vol_ratio:.2f}")
                print(f" Volume factor counter: {vol_fact_c}")
                if vol_fact_c > max_vol_fact_c:
                    max_vol_fact_c = vol_fact_c
                vol_fact_c = 0
                print(f" Maximum volume factor counter: {max_vol_fact_c}")
                print(f" Radio a: {radii[0]:.2f}; b: {radii[1]:.2f}; c: {radii[2]:.2f}")
                print(" -------------------------------------------------------------------------")    
            
            self.im[np.where(ellipsoid == 1)] = index
            poros2 = self.calculate_porosity(self.im)
            if poros1 > poros2:
                ell_not_cre_counter = 0
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['x'].append(diams_list[0])
                self.statistics['diameters']['y'].append(diams_list[1])
                self.statistics['diameters']['z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                if angle_tolerance_flag == True and angles_range_flag == False:
                    self.statistics['angles']['theta_x'].append(angles[0])
                    self.statistics['angles']['theta_y'].append(angles[1])
                    self.statistics['angles']['theta_z'].append(angles[2])
                """ A counter of the number of balls generated"""
                n_balls += 1
                
            # if n_balls % 100 == 0:
            #     print(" Ellipsoids created: ", n_balls)
            #     print(f" Porosity: {self.statistics['porosity']:.4f}")
            #     print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
            #     print("")
            # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
            # if n_balls >= 10:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            #     break     
        
        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)

    def add_ellipsoids_gravity_cfd_individual(self, target_porosity: float, 
                                              diams: npt.NDArray[np.float64],
                                              cfd: npt.NDArray[np.float64], 
                                              ratios: List[float] = None, 
                                              overlap: float = 0.0,
                                              electrode_generation_mode: str = 'extended',
                                              angle_tolerance_flag: bool = True,
                                              angle_tolerance: int = 5, 
                                              angles_range_flag: bool = True,
                                              theta_x: int = 0,
                                              theta_y: int = 0,
                                              theta_z: int = 0,
                                              mode: int = 0,
                                              max_tries: int = 100,
                                              axis: int = 0,
                                              ):
        """
        Generates and adds ellipsoids to the structure while keeping track of porosity.
        Each generated ellipsoid is assigned a unique label number.
        
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible ellipsoid diameters for the three axes.
        - cfd: Cumulative Frequency Distribution for each axis.
        - ratios: Proportions for different groups of ellipsoids (must sum to 1.0).
        - overlap: Allowed overlap between ellipsoids (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angle_tolerance_flag: Whether to enable random orientation within a specified range.
        - angle_tolerance: Range for random orientation if `angle_tolerance_flag` is enabled.
        - theta_x, theta_y, theta_z: Fixed orientation angles if `angle_tolerance_flag` is disabled.
        - mode: If set, enables random size selection based on ratios.
        """
        
        # Check if the image (structure) has been defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Validate the ratios, if provided.
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')  # Ensure ratios sum to 1.
            else:
                cumsum = np.cumsum(ratios)  # Compute cumulative sum of ratios.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []  # Initialize diameter lists.
        else:
            # If no ratios are provided, initialize empty lists for diameter and angle statistics.
            cumsum = [1.0]
            self.statistics['diameters']['x'] = []
            self.statistics['diameters']['y'] = []
            self.statistics['diameters']['z'] = []
            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
    
        n_balls = 0  # Counter for generated ellipsoids.
        ell_not_cre_counter = 0  # Counter for unsuccessful ellipsoid creations.
        vol_ratio = 1.00  # Initial volume ratio threshold.
        max_vol_fact_c = 0  # Tracks the maximum volume factor counter.
        vol_fact_c = 0
        tries = 0  # Counter for tracking consecutive failed placement attempts.
    
        print(f" Volume ratio: {vol_ratio}")
    
        # Loop until the target porosity is reached.
        while target_porosity < self.statistics['porosity']:
            if mode:
                # Randomly select a mode based on cumulative sums.
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
            
            # Calculate the current porosity of the structure.
            poros1 = self.calculate_porosity(self.im)
    
            # Initialize lists for diameters and radii.
            diams_list = [[], [], []]
            radii = []
            radii_aux = []
            
            # Generate random factors for ellipsoid dimensions.
            factor_y = random.uniform(0.5, 0.75)
            factor_z = random.uniform(0.15, 0.4)
            diams_sph = random.uniform(6.5, 14.5)  # Random sphere diameter.
            vols = 4 * np.pi * (diams_sph / 2)**3 / 3  # Volume of the sphere.
            diams_x = 2 * (3 * vols / (4 * np.pi * factor_y * factor_z))**(1/3)  # Calculate x-diameter.
            diams_y = factor_y * diams_x  # Calculate y-diameter based on factor.
            diams_z = factor_z * diams_x  # Calculate z-diameter based on factor.            
     
            # Calculate radii in voxel units.
            radii = [diams_x * 0.5 / self.voxel,
                      diams_y * 0.5 / self.voxel,
                      diams_z * 0.5 / self.voxel]
            diams_list[0] = diams_x
            diams_list[1] = diams_y
            diams_list[2] = diams_z
    
            # Determine the index of the diameter to sample based on current porosity.
            if poros1 > 2 * target_porosity:
                idmin = random.randint(int(len(diams[0]) * 0.5), len(diams[0]) - 1)
            elif poros1 > 1.5 * target_porosity and poros1 < 2 * target_porosity:
                idmin = random.randint(int(len(diams[0]) * 0.1), int(len(diams[0]) * 0.4))
            else:
                idmin = random.randint(int(len(diams[0]) * 0.075), len(diams[0]) - 1)
    
            # Retrieve diameters and compute radii.
            # rnd2 = np.random.random_sample()
            for i in range(3):
                # idmin = (np.abs(cfd[mode] - rnd2)).argmin()
                diams_list.append(diams[i][idmin])  # Append diameter to the list.
                radii_aux.append(round(diams_list[i] * 0.5 / self.voxel + 0.5))  # Adjust radius.
                radii.append(diams_list[i] * 0.5 / self.voxel)  # Append radius in voxel units.
    
            el_vol = 4 / 3 * np.pi * radii[0] * radii[1] * radii[2]  # Calculate ellipsoid volume.
    
            # Check angle tolerances and select random angles for rotation if enabled.
            if angle_tolerance_flag:
                angle_tolerance = angle_tolerance
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                theta_x = random.choice(chosen_range_x)  # Select random x-angle.
                theta_y = random.choice(chosen_range_y)  # Select random y-angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                    
            # Check if angle ranges are specified and select angles accordingly.
            if angles_range_flag and not angle_tolerance_flag:        
                # Define angle ranges for x and y rotations.
                theta_x_values = [-10, -5, 0, 5, 10]
                theta_y_values = [-10, -5, 0, 5, 10]
                combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))  # Generate all combinations.
                combination_indices = {combination: index + 2 for index, combination in enumerate(combinations)}
                
                self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]  # Store x angles.
                self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]  # Store y angles.
                self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]  # Zero z angles.
    
                theta_x = random.choice(theta_x_values)  # Randomly choose x angle.
                theta_y = random.choice(theta_y_values)  # Randomly choose y angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(-5, 5)
                
                index = combination_indices[(theta_x, theta_y)]  # Assign index based on angle combination.
            else:    
                # If no angle range is specified, use provided angles.
                theta_x = theta_x
                theta_y = theta_y
                theta_z = theta_z
                index = n_balls + 2  # Assign index for the ellipsoid.
    
            # Determine protrusion based on maximum radius and overlap.
            clearanceA = -1 * int(diams_list[0] * overlap + 0.5)  # Set the clearance based on overlap.
            clearanceB = -1 * int(diams_list[1] * overlap + 0.5)  # Set the clearance based on overlap.
            clearanceC = -1 * int(diams_list[2] * overlap + 0.5)  # Set the clearance based on overlap.

            if abs(clearanceA) > radii[0]:
                clearanceA = np.sign(clearanceA) * radii[0]
            if abs(clearanceB) > radii[1]:
                clearanceB = np.sign(clearanceB) * radii[1]
            if abs(clearanceC) > radii[2]:
                clearanceC = np.sign(clearanceC) * radii[2]
                     
            try:
                # Generate a new sphere using pseudo-gravity packing.
                ellipsoid, angles = 1 * ps.generators.pseudo_gravity_packing_ellipsoids(self.im==0, 
                                                                                    rA=round(radii[0]), rB=round(radii[1]), rC=round(radii[2]), 
                                                                                    rotation_angles=False,
                                                                                    theta_x=np.radians(theta_x), 
                                                                                    theta_y=np.radians(theta_y), 
                                                                                    theta_z=np.radians(theta_z),
                                                                                    clearanceA=clearanceA, clearanceB=clearanceB, clearanceC=clearanceC,
                                                                                    axis=axis, maxiter=1, edges='extended')
                # print(" Ellipsoid generated...")
                # Update the structure's image with the new sphere.
                self.im[np.nonzero(ellipsoid)] = 0  # Reset previous sphere positions to avoid overlap.
                self.im += ellipsoid.astype(np.uint16)  # Add the new sphere to the structure.
                angles = [theta_x, theta_y, theta_z]
                
            except ValueError:
                # Handle cases where a sphere could not be placed successfully.
                tries += 1  # Increment the failure counter.
                # print( f" Try nº {tries}")
                if tries >= max_tries:  # Stop if the maximum number of attempts is reached.
                    print(" Value error. Nº of tries: ", tries)
                    break
                # if time.time() - start_time > 10:  # Stop if the time limit is exceeded.
                #     print("Time exceeded.")
                #     break
                continue  # Skip to the next iteration if placement fails.
    
            print(" Ellipsoid kept")
            print(f" Particle id: {index}")
            print(f" Original volume: {el_vol:.2f}")
            print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
            print(f" Volume factor: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
            print(f" Porosity: {poros1*100:.2f} %")
            print(f" Volume ratio: {vol_ratio:.2f}")
            print(f" Volume factor counter: {vol_fact_c}")
            if vol_fact_c > max_vol_fact_c:
                max_vol_fact_c = vol_fact_c
            vol_fact_c = 0
            print(f" Maximum volume factor counter: {max_vol_fact_c}")
            print(f" Radio a: {radii[0]:.2f}; b: {radii[1]:.2f}; c: {radii[2]:.2f}")
            print(f" Angle alpha: {angles[0]:.2f}; beta: {angles[1]:.2f}; gamma: {angles[2]:.2f}")
            print(f" Clearances: a = {clearanceA}, b = {clearanceB}, c = {clearanceC}")
            print(" -------------------------------------------------------------------------")    
            
            self.im[np.where(ellipsoid == 1)] = index
            poros2 = self.calculate_porosity(self.im)
            if poros1 > poros2:
                ell_not_cre_counter = 0
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['x'].append(diams_list[0])
                self.statistics['diameters']['y'].append(diams_list[1])
                self.statistics['diameters']['z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                if angle_tolerance_flag == True and angles_range_flag == False:
                    self.statistics['angles']['theta_x'].append(angles[0])
                    self.statistics['angles']['theta_y'].append(angles[1])
                    self.statistics['angles']['theta_z'].append(angles[2])
                """ A counter of the number of balls generated"""
                n_balls += 1
                tries = 0
                
            if n_balls % 1 == 0:
                print(" Ellipsoids created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
                print("")
            # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
            # if n_balls >= 10:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            #     break     
    
        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)

    def add_ellipsoids_gravity_cfd_counted_individual(self, target_porosity: float, 
                                                      diams: npt.NDArray[np.float64],
                                                      cfd: npt.NDArray[np.float64], 
                                                      ratios: List[float] = None, 
                                                      overlap: float = 0.0,
                                                      electrode_generation_mode: str = 'extended',
                                                      angle_tolerance_flag: bool = True,
                                                      angle_tolerance: int = 5, 
                                                      angles_range_flag: bool = True,
                                                      theta_x: int = 0,
                                                      theta_y: int = 0,
                                                      theta_z: int = 0,
                                                      mode: int = 0,
                                                      max_tries: int = 100,
                                                      axis: int = 0,
                                                      ):
        """
        Generates and adds ellipsoids to the structure while keeping track of porosity.
        Each generated ellipsoid is assigned a unique label number.
        
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diams: Array of possible ellipsoid diameters for the three axes.
        - cfd: Cumulative Frequency Distribution for each axis.
        - ratios: Proportions for different groups of ellipsoids (must sum to 1.0).
        - overlap: Allowed overlap between ellipsoids (0.0 means no overlap).
        - electrode_generation_mode: Mode for RSA generation (default is 'extended').
        - angle_tolerance_flag: Whether to enable random orientation within a specified range.
        - angle_tolerance: Range for random orientation if `angle_tolerance_flag` is enabled.
        - theta_x, theta_y, theta_z: Fixed orientation angles if `angle_tolerance_flag` is disabled.
        - mode: If set, enables random size selection based on ratios.
        """
        
        # Check if the image (structure) has been defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        # Validate the ratios, if provided.
        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')  # Ensure ratios sum to 1.
            else:
                cumsum = np.cumsum(ratios)  # Compute cumulative sum of ratios.
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []  # Initialize diameter lists.
        else:
            # If no ratios are provided, initialize empty lists for diameter and angle statistics.
            # cumsum = [1.0]
            self.statistics['diameters']['x'] = []
            self.statistics['diameters']['y'] = []
            self.statistics['diameters']['z'] = []
            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
    
        n_balls = 0  # Counter for generated ellipsoids.
        # ell_not_cre_counter = 0  # Counter for unsuccessful ellipsoid creations.
        vol_ratio = 1.00  # Initial volume ratio threshold.
        # max_vol_fact_c = 0  # Tracks the maximum volume factor counter.
        # vol_fact_c = 0
        # tries = 0  # Counter for tracking consecutive failed placement attempts.
    
        print(f" Volume ratio: {vol_ratio}")
    
        # Loop until the target porosity is reached.
        particle_diameters = self._estimate_number(target_porosity, diams, cfd, 
                                                   ratios, particle='ellipsoid')
        # while any(sublist for sublist in particle_diameters) or target_porosity < self.statistics['porosity']:
        while target_porosity < self.statistics['porosity']:
            if particle_diameters[0]:
                diams_x, diams_y, diams_z = particle_diameters[0].pop(0)
            else:
                print(" Warning: No more elements to pop from particle_diameters!")
                break 
            
            # Calculate the current porosity of the structure.
            poros1 = self.calculate_porosity(self.im)
    
            # Initialize lists for diameters and radii.
            diams_list = [[], [], []]
            radii = []
            
            # Calculate radii in voxel units.
            radii = [diams_x * 0.5 / self.voxel,
                      diams_y * 0.5 / self.voxel,
                      diams_z * 0.5 / self.voxel]
            diams_list[0] = diams_x
            diams_list[1] = diams_y
            diams_list[2] = diams_z
            
    
            # Check angle tolerances and select random angles for rotation if enabled.
            if angle_tolerance_flag:
                angle_tolerance = angle_tolerance
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                theta_x = random.choice(chosen_range_x)  # Select random x-angle.
                theta_y = random.choice(chosen_range_y)  # Select random y-angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(0, 360)
                    
            # Check if angle ranges are specified and select angles accordingly.
            if angles_range_flag and not angle_tolerance_flag:        
                # Define angle ranges for x and y rotations.
                theta_x_values = [-10, -5, 0, 5, 10]
                theta_y_values = [-10, -5, 0, 5, 10]
                combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))  # Generate all combinations.
                combination_indices = {combination: index + 2 for index, combination in enumerate(combinations)}
                
                self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]  # Store x angles.
                self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]  # Store y angles.
                self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]  # Zero z angles.
    
                theta_x = random.choice(theta_x_values)  # Randomly choose x angle.
                theta_y = random.choice(theta_y_values)  # Randomly choose y angle.
                # Generate a random z-angle for rotation.
                theta_z = random.uniform(0, 360)
                
                index = combination_indices[(theta_x, theta_y)]  # Assign index based on angle combination.
            else:    
                # If no angle range is specified, use provided angles.
                theta_x = theta_x
                theta_y = theta_y
                theta_z = theta_z
                index = n_balls + 2  # Assign index for the ellipsoid.
    
            # Determine protrusion based on maximum radius and overlap.
            clearanceA = -1 * int(diams_list[0] * overlap + 0.5)  # Set the clearance based on overlap.
            clearanceB = -1 * int(diams_list[1] * overlap + 0.5)  # Set the clearance based on overlap.
            clearanceC = -1 * int(diams_list[2] * overlap + 0.5)  # Set the clearance based on overlap.

            try:
                # Generate a new sphere using pseudo-gravity packing.
                ellipsoid, angles = 1 * ps.generators.pseudo_gravity_packing_ellipsoids(self.im==0, 
                                                                                        rA=round(radii[0] + clearanceA), 
                                                                                        rB=round(radii[1] + clearanceB), 
                                                                                        rC=round(radii[2] + clearanceC), 
                                                                                        rotation_angles=False,
                                                                                        theta_x=np.radians(theta_x), 
                                                                                        theta_y=np.radians(theta_y), 
                                                                                        theta_z=np.radians(theta_z),
                                                                                        clearanceA=clearanceA,
                                                                                        clearanceB=clearanceB,
                                                                                        clearanceC=clearanceC,
                                                                                        axis=axis,
                                                                                        maxiter=1, 
                                                                                        edges='extended'
                                                                                        )
                # Update the structure's image with the new sphere.
                self.im[np.nonzero(ellipsoid)] = 0  # Reset previous sphere positions to avoid overlap.
                self.im += ellipsoid.astype(np.uint16)  # Add the new sphere to the structure.
                angles = [theta_x, theta_y, theta_z]
                
            except ValueError:
                # print(" Ellipsoid insertion failed. ")
                continue  # Skip to the next iteration if placement fails.
    
            # print(" Ellipsoid kept")
            # print(f" Particle id: {index}")
            # print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
            # print(f" Porosity: {poros1*100:.2f} %")
            # print(f" Volume ratio: {vol_ratio:.2f}")
            # print(f" Volume factor counter: {vol_fact_c}")
            # if vol_fact_c > max_vol_fact_c:
            #     max_vol_fact_c = vol_fact_c
            # vol_fact_c = 0
            # print(f" Maximum volume factor counter: {max_vol_fact_c}")
            # print(f" Radio a: {radii[0]:.2f}; b: {radii[1]:.2f}; c: {radii[2]:.2f}")
            # print(f" Angle alpha: {angles[0]:.2f}; beta: {angles[1]:.2f}; gamma: {angles[2]:.2f}")
            # print(" -------------------------------------------------------------------------")    
            
            self.im[np.where(ellipsoid == 1)] = index
            poros2 = self.calculate_porosity(self.im)
            if poros1 > poros2:
                # ell_not_cre_counter = 0
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['x'].append(diams_list[0])
                self.statistics['diameters']['y'].append(diams_list[1])
                self.statistics['diameters']['z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                if angle_tolerance_flag == True and angles_range_flag == False:
                    self.statistics['angles']['theta_x'].append(angles[0])
                    self.statistics['angles']['theta_y'].append(angles[1])
                    self.statistics['angles']['theta_z'].append(angles[2])
                """ A counter of the number of balls generated"""
                n_balls += 1
                # tries = 0
                
            if n_balls % 1 == 0:
                print(" Ellipsoids created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
                print("")
            # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
            # if n_balls >= 10:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            #     break     
    
        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)

    def add_mixed_geom_rsa_cfd_individual(self, target_porosity: float, 
                                           diams: npt.NDArray[np.float64],
                                           cfd: npt.NDArray[np.float64], 
                                           ratios: List[float] = None, 
                                           overlap: float = 0.0,
                                           electrode_generation_mode: str = 'extended',
                                           angle_tolerance_flag: bool = True,
                                           angle_tolerance: int = 5, 
                                           angles_range_flag: bool = True,
                                           theta_x: int = 0,
                                           theta_y: int = 0,
                                           theta_z: int = 0,
                                           mode: int = 0):
        
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')

        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                cumsum = np.cumsum(ratios)
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []

        else:
            cumsum = [1.0]
            self.statistics['diameters']['ellipsoid_x'] = []
            self.statistics['diameters']['ellipsoid_y'] = []
            self.statistics['diameters']['ellipsoid_z'] = []
            self.statistics['diameters']['sphere'] = []

            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
                         
        im_sp = np.zeros([75, 75, 45], dtype=np.uint16)
        n_balls = 0   
        ell_not_cre_counter = 0
        vol_ratio = 0.95
        print(f" Volume ratio: {vol_ratio}")
        while target_porosity < self.statistics['porosity']:
            if mode:
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]

            rnd2 = np.random.random_sample()
            idmin = (np.abs(cfd[mode] - rnd2)).argmin()
            diam = diams[0][idmin]
            rad = int(diam * 0.5 / self.voxel + 0.5)
            protrusion = int(diam * overlap + 0.5)

            poros1 = self.calculate_porosity(im_sp)
            
            index = np.max(im_sp) + 1
            if index > n_balls+2:
                im_sp[im_sp > n_balls] = n_balls
                index = n_balls + 2
            if n_balls == 0 and index == 1:
                index = 2
            try:
                ball = ps.generators.rsa(im_or_shape=im_sp, r=rad, n_max=1,
                                         mode='extended',
                                         protrusion=protrusion)
            except:
                continue
            
            im_sp[np.where(ball == 1)] = index
            poros2 = self.calculate_porosity(im_sp)

            if poros1 > poros2:
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['sphere'].append(diam)
                # self.statistics['diameters']['Total'].append(diam)
                # if angles:
                    # self.statistics['angles'].append(random.randint(0, 2*np.pi))    
            else:
                continue

            """ A counter of the number of balls generated"""
            n_balls += 1
            if n_balls % 5 == 0:
                print(" Balls created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Diameter of the ball: {self.statistics['diameters']['sphere'][-1]:.2f} um")
                print("")
            
        self.im[:,:,:45] = im_sp
        self.statistics['porosity'] = self.calculate_porosity(self.im)
        while target_porosity < self.statistics['porosity']:
            if mode:
                rnd1 = np.random.random_sample()
                mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
            
            poros1 = self.calculate_porosity(self.im)
            
            diams_list = [[], [], []]
            radii = []
            # radii_aux = []
            
            # factor_y = random.uniform(0.5, 0.75)
            # factor_z = random.uniform(0.05, 0.2)
            # diams_sph = random.uniform(7.5, 20)
            # vols = 4*np.pi*(diams_sph/2)**3/3
            # diams_x = 2*(3*vols/(4*np.pi*factor_y*factor_z))**(1/3)
            # diams_y = factor_y*diams_x
            # diams_z = factor_z*diams_x
            # vols_ellipsoid = 4*np.pi*(diams_x/2)*(diams_y/2)*(diams_z/2)/3
            # # Check volume similarity for spheres and ellipsoids                
            # vol_similarity = np.mean(vols/vols_ellipsoid)
            # print(f" Avg. V_sph/V_ell: {vol_similarity:.2f}")
            # radii = [diams_x * 0.5 / self.voxel,
            #           diams_y * 0.5 / self.voxel,
            #           diams_z * 0.5 / self.voxel
            #     ]
            
            # diams_list[0] = diams_x
            # diams_list[1] = diams_y
            # diams_list[2] = diams_z
 
            # if poros1 > 2*target_porosity:
            #     idmin = random.randint(int(len(diams[1])*0.5), len(diams[1])-1)
            # elif poros1 > 1.5*target_porosity and poros1 < 2*target_porosity:
            #     idmin = random.randint(int(len(diams[1])*0.1), int(len(diams[1])*0.4))
            # else:
            #     idmin = random.randint(int(len(diams[1])*0.075), len(diams[1])-1)
            rnd2 = np.random.random_sample()
            for i in range(1,4):
                idmin = (np.abs(cfd[mode] - rnd2)).argmin()
                diams_list.append(diams[i][idmin])
                # radii_aux.append(round(diams_list[i-1] * 0.5 / self.voxel + 0.5))
                radii.append(diams[i][idmin] * 0.5 / self.voxel)
 
            # el_vol = 4/3*np.pi*radii[0]*radii[1]*radii[2] 
 
            if angle_tolerance_flag:
                angle_tolerance = angle_tolerance
                chosen_range_x = range(-angle_tolerance, angle_tolerance)
                chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                theta_x = random.choice(chosen_range_x)
                theta_y = random.choice(chosen_range_y)
                index = np.max(self.im) + 1
                if index > n_balls+2:
                    self.im[self.im > n_balls] = n_balls
                    index = n_balls + 1
                if n_balls == 0 and index == 1:
                    index = 2
                    
            if angles_range_flag == True and angle_tolerance_flag == False:        
                theta_x_values = [-10, -5, 0, 5, 10]
                theta_y_values = [-10, -5, 0, 5, 10]
                # theta_y_values = [0]
                combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))
                combination_indices = {combination: index+2 for index, combination in enumerate(combinations)}
                
                self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]
                self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]
                self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]
 
                theta_x = random.choice(theta_x_values)
                theta_y = random.choice(theta_y_values)
 
                index = combination_indices[(theta_x, theta_y)]
            else:    
                theta_x = theta_x
                theta_y = theta_y
                theta_z = theta_z
                index = n_balls + 2
            theta_z = 0
            # theta_z = random.uniform(-10, 10)
 
            protrusion = int(max(radii)*2 * overlap + 0.5)
            try:
                ellipsoid, angles = ps.generators.rsa_ellipsoids(im_or_shape=self.im, rA=round(radii[0]), rB=round(radii[1]), rC=round(radii[2]), 
                                                                  n_max=1, mode=electrode_generation_mode, protrusion=protrusion, 
                                                                  smooth=True, rotation_angles=True, theta_x=np.radians(theta_x), 
                                                                  theta_y=np.radians(theta_y), theta_z=np.radians(theta_z))                
            except:
                ell_not_cre_counter += 1
                if ell_not_cre_counter > 10000:
                    print(" Structure generation stopped.")
                    break
                continue
            
            location = np.where(ellipsoid==1)
            [x, y, z] = [np.mean(loc) for loc in location]
            # print(f' Location in thickness: {z}. Thickness: {self.im.shape[2]}')
            if z <= 30:
                index = 0

            # if (len(np.where(ellipsoid==1)[0]) < vol_ratio*el_vol):
            #     # and not (((np.any(np.where(ellipsoid==1)[0]) == True or np.any(np.where(ellipsoid==1)[1]) == True) 
            #     #          and len(np.where(ellipsoid==1)[0]) < 0.99*el_vol)
            #     # and not (np.any(np.where(ellipsoid==1)[2]) == True 
            #     #          and len(np.where(ellipsoid==1)[0]) < (vol_ratio-0.05)*el_vol))):
            # # if len(np.where(ellipsoid==1)[0]) < 0.8*el_vol or len(np.where(ellipsoid==1)[0]) > 1.2*el_vol:
            #     # print(" Too much overlap in the ellipsoid.")
            #     # print(" Ellipsoid discarded.")
            #     # print(f" Original volume: {el_vol:.2f}")
            #     # print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
            #     # print(f" Volume factor {index}: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
            #     # print(f" Porosity: {poros1*100:.2f} %")
            #     index = 0
            #     vol_fact_c +=1
            #     if vol_fact_c >= 500:
            #         vol_ratio -= 0.025
            #         print(f" Volume facor overcome. {vol_ratio:.2f}, {vol_fact_c}")
            #     # print(" -------------------------------------------------------------------------")
            # else:
            #     print(" Ellipsoid kept")
            #     print(f" Particle id: {index}")
            #     print(f" Original volume: {el_vol:.2f}")
            #     print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
            #     print(f" Volume factor: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
            #     print(f" Porosity: {poros1*100:.2f} %")
            #     print(f" Volume ratio: {vol_ratio:.2f}")
            #     print(f" Volume factor counter: {vol_fact_c}")
            #     if vol_fact_c > max_vol_fact_c:
            #         max_vol_fact_c = vol_fact_c
            #     vol_fact_c = 0
            #     print(f" Maximum volume factor counter: {max_vol_fact_c}")
            #     print(f" Radio a: {radii[0]:.2f}")
            #     print(" -------------------------------------------------------------------------")    
 
            if angles_range_flag:
                self.im[np.where(ellipsoid == 1)] = index
            else:
                self.im[np.where(ellipsoid == 1)] = index
            poros2 = self.calculate_porosity(self.im)
            
            if poros1 > poros2:
                ell_not_cre_counter = 0
                self.statistics['porosity'] = poros2
                self.statistics['diameters']['ellipsoid_x'].append(diams_list[0])
                self.statistics['diameters']['ellipsoid_y'].append(diams_list[1])
                self.statistics['diameters']['ellipsoid_z'].append(diams_list[2])
                self.statistics['diameters']['Total'].append(diams_list[0])
                if angle_tolerance_flag == True and angles_range_flag == False:
                    self.statistics['angles']['theta_x'].append(angles[0])
                    self.statistics['angles']['theta_y'].append(angles[1])
                    self.statistics['angles']['theta_z'].append(angles[2])
                """ A counter of the number of balls generated"""
                n_balls += 1
                
            if n_balls % 100 == 0:
                print(" Ellipsoids created: ", n_balls)
                print(f" Porosity: {self.statistics['porosity']:.4f}")
                print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
                print("")
            # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
            # if n_balls >= 10:
            # if self.statistics['porosity'] <= 1.025*target_porosity:
            #     break
    
        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)
    
    def add_mixed_geom_rsa_cfd_individual_2(self, target_porosity: float, 
                                           diams: npt.NDArray[np.float64],
                                           cfd: npt.NDArray[np.float64], 
                                           ratios: List[float] = None, 
                                           overlap: float = 0.0,
                                           electrode_generation_mode: str = 'extended',
                                           angle_tolerance_flag: bool = True,
                                           angle_tolerance: int = 5, 
                                           angles_range_flag: bool = True,
                                           theta_x: int = 0,
                                           theta_y: int = 0,
                                           theta_z: int = 0,
                                           mode: int = 0):
        
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')

        if ratios is not None:
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError('Ratios must sum to 1.0')
            else:
                cumsum = np.cumsum(ratios)
                for i in range(len(ratios)):
                    self.statistics['diameters'][str(i + 1)] = []

        else:
            cumsum = [1.0]
            self.statistics['diameters']['ellipsoid_x'] = []
            self.statistics['diameters']['ellipsoid_y'] = []
            self.statistics['diameters']['ellipsoid_z'] = []
            self.statistics['diameters']['sphere'] = []

            self.statistics['angles']['theta_x'] = []
            self.statistics['angles']['theta_y'] = []
            self.statistics['angles']['theta_z'] = []
                         
        # im_sp = np.zeros([75, 75, 45], dtype=np.uint16)
        n_balls = 0   
        ell_not_cre_counter = 0
        # vol_fact_c = 0
        vol_ratio = 0.95
        # max_vol_fact_c = 0
        print(f" Volume ratio: {vol_ratio}")
        while target_porosity < self.statistics['porosity']:
            rnd_geometry = random.randint(0, 1)
            if rnd_geometry == 0:
                if mode:
                    rnd1 = np.random.random_sample()
                    mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
                
                rnd2 = np.random.random_sample()
                idmin = (np.abs(cfd[mode] - rnd2)).argmin()
                diam = diams[0][idmin]
                rad = int(diam * 0.5 / self.voxel + 0.5)
                protrusion = int(diam * overlap + 0.5)
    
                poros1 = self.calculate_porosity(self.im)
                
                index = np.max(self.im) + 1
                if index > n_balls+2:
                    self.im[self.im > n_balls] = n_balls
                    index = n_balls + 2
                if n_balls == 0 and index == 1:
                    index = 2
                try:
                    ball = ps.generators.rsa(im_or_shape=self.im, r=rad, n_max=1,
                                             mode='extended',
                                             protrusion=protrusion)
                except:
                    continue
                
                location = np.where(ball==1)
                [x, y, z] = [np.mean(loc) for loc in location]
                # print(f' Location in thickness: {z}. Thickness: {self.im.shape[2]}')
                if z >= 40:
                    index = 0
                
                self.im[np.where(ball == 1)] = index
                poros2 = self.calculate_porosity(self.im)
    
                if poros1 > poros2:
                    self.statistics['porosity'] = poros2
                    self.statistics['diameters']['sphere'].append(diam)
                    # self.statistics['diameters']['Total'].append(diam)
                    # if angles:
                        # self.statistics['angles'].append(random.randint(0, 2*np.pi))    
                else:
                    continue
    
                """ A counter of the number of balls generated"""
                n_balls += 1
                if n_balls % 5 == 0:
                    print(" Balls created: ", n_balls)
                    print(f" Porosity: {self.statistics['porosity']:.4f}")
                    print(f" Diameter of the ball: {self.statistics['diameters']['sphere'][-1]:.2f} um")
                    print("")
                    
            if rnd_geometry == 1:
                if mode:
                    rnd1 = np.random.random_sample()
                    mode = [i + 1 if cumsum[i] < rnd1 < cumsum[i + 1] else 0 for i in range(len(cumsum) - 1)][0]
                
                poros1 = self.calculate_porosity(self.im)
                
                diams_list = [[], [], []]
                radii = []
                # radii_aux = []
                
                # factor_y = random.uniform(0.5, 0.75)
                # factor_z = random.uniform(0.05, 0.2)
                # diams_sph = random.uniform(7.5, 20)
                # vols = 4*np.pi*(diams_sph/2)**3/3
                # diams_x = 2*(3*vols/(4*np.pi*factor_y*factor_z))**(1/3)
                # diams_y = factor_y*diams_x
                # diams_z = factor_z*diams_x
                # vols_ellipsoid = 4*np.pi*(diams_x/2)*(diams_y/2)*(diams_z/2)/3
                # # Check volume similarity for spheres and ellipsoids                
                # vol_similarity = np.mean(vols/vols_ellipsoid)
                # print(f" Avg. V_sph/V_ell: {vol_similarity:.2f}")
                # radii = [diams_x * 0.5 / self.voxel,
                #           diams_y * 0.5 / self.voxel,
                #           diams_z * 0.5 / self.voxel
                #     ]
                
                # diams_list[0] = diams_x
                # diams_list[1] = diams_y
                # diams_list[2] = diams_z
     
                if poros1 > 2*target_porosity:
                    idmin = random.randint(int(len(diams[1])*0.5), len(diams[1])-1)
                elif poros1 > 1.5*target_porosity and poros1 < 2*target_porosity:
                    idmin = random.randint(int(len(diams[1])*0.1), int(len(diams[1])*0.4))
                else:
                    idmin = random.randint(int(len(diams[1])*0.075), len(diams[1])-1)
                rnd2 = np.random.random_sample()
                for i in range(1,4):
                    # idmin = (np.abs(cfd[mode] - rnd2)).argmin()
                    diams_list.append(diams[i][idmin])
                    # radii_aux.append(round(diams_list[i-1] * 0.5 / self.voxel + 0.5))
                    radii.append(diams[i][idmin] * 0.5 / self.voxel)
     
                # el_vol = 4/3*np.pi*radii[0]*radii[1]*radii[2] 
     
                if angle_tolerance_flag:
                    angle_tolerance = angle_tolerance
                    chosen_range_x = range(-angle_tolerance, angle_tolerance)
                    chosen_range_y = range(-angle_tolerance, angle_tolerance)               
                    theta_x = random.choice(chosen_range_x)
                    theta_y = random.choice(chosen_range_y)
                    index = np.max(self.im) + 1
                    if index > n_balls+2:
                        self.im[self.im > n_balls] = n_balls
                        index = n_balls + 1
                    if n_balls == 0 and index == 1:
                        index = 2
                        
                if angles_range_flag == True and angle_tolerance_flag == False:        
                    theta_x_values = [-10, -5, 0, 5, 10]
                    theta_y_values = [-10, -5, 0, 5, 10]
                    # theta_y_values = [0]
                    combinations = list(itertools.product(theta_x_values, theta_y_values, repeat=1))
                    combination_indices = {combination: index+2 for index, combination in enumerate(combinations)}
                    
                    self.statistics['angles']['theta_x'] = [tup[0] for tup in combinations]
                    self.statistics['angles']['theta_y'] = [tup[1] for tup in combinations]
                    self.statistics['angles']['theta_z'] = [0 for a in range(len(combinations))]
     
                    theta_x = random.choice(theta_x_values)
                    theta_y = random.choice(theta_y_values)
     
                    index = combination_indices[(theta_x, theta_y)]
                else:    
                    theta_x = theta_x
                    theta_y = theta_y
                    theta_z = theta_z
                    index = n_balls + 2
                theta_z = 0
                # theta_z = random.uniform(-10, 10)
     
                protrusion = int(max(radii)*2 * overlap + 0.5)
                try:
                    ellipsoid, angles = ps.generators.rsa_ellipsoids(im_or_shape=self.im, rA=round(radii[0]), rB=round(radii[1]), rC=round(radii[2]), 
                                                                      n_max=1, mode=electrode_generation_mode, protrusion=protrusion, 
                                                                      smooth=True, rotation_angles=True, theta_x=np.radians(theta_x), 
                                                                      theta_y=np.radians(theta_y), theta_z=np.radians(theta_z))                
                except:
                    ell_not_cre_counter += 1
                    if ell_not_cre_counter > 10000:
                        print(" Structure generation stopped.")
                        break
                    continue
                
                location = np.where(ellipsoid==1)
                [x, y, z] = [np.mean(loc) for loc in location]
                # print(f' Location in thickness: {z}. Thickness: {self.im.shape[2]}')
                if z <= 30:
                    index = 0

                # if (len(np.where(ellipsoid==1)[0]) < vol_ratio*el_vol):
                #     # and not (((np.any(np.where(ellipsoid==1)[0]) == True or np.any(np.where(ellipsoid==1)[1]) == True) 
                #     #          and len(np.where(ellipsoid==1)[0]) < 0.99*el_vol)
                #     # and not (np.any(np.where(ellipsoid==1)[2]) == True 
                #     #          and len(np.where(ellipsoid==1)[0]) < (vol_ratio-0.05)*el_vol))):
                # # if len(np.where(ellipsoid==1)[0]) < 0.8*el_vol or len(np.where(ellipsoid==1)[0]) > 1.2*el_vol:
                #     # print(" Too much overlap in the ellipsoid.")
                #     # print(" Ellipsoid discarded.")
                #     # print(f" Original volume: {el_vol:.2f}")
                #     # print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
                #     # print(f" Volume factor {index}: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
                #     # print(f" Porosity: {poros1*100:.2f} %")
                #     index = 0
                #     vol_fact_c +=1
                #     if vol_fact_c >= 500:
                #         vol_ratio -= 0.025
                #         print(f" Volume facor overcome. {vol_ratio:.2f}, {vol_fact_c}")
                #     # print(" -------------------------------------------------------------------------")
                # else:
                #     print(" Ellipsoid kept")
                #     print(f" Particle id: {index}")
                #     print(f" Original volume: {el_vol:.2f}")
                #     print(f" Real volume: {len(np.where(ellipsoid==1)[0])}")
                #     print(f" Volume factor: {len(np.where(ellipsoid==1)[0])/el_vol*100:.2f} %")
                #     print(f" Porosity: {poros1*100:.2f} %")
                #     print(f" Volume ratio: {vol_ratio:.2f}")
                #     print(f" Volume factor counter: {vol_fact_c}")
                #     if vol_fact_c > max_vol_fact_c:
                #         max_vol_fact_c = vol_fact_c
                #     vol_fact_c = 0
                #     print(f" Maximum volume factor counter: {max_vol_fact_c}")
                #     print(f" Radio a: {radii[0]:.2f}")
                #     print(" -------------------------------------------------------------------------")    
     
                if angles_range_flag:
                    self.im[np.where(ellipsoid == 1)] = index
                else:
                    self.im[np.where(ellipsoid == 1)] = index
                poros2 = self.calculate_porosity(self.im)
                
                if poros1 > poros2:
                    ell_not_cre_counter = 0
                    self.statistics['porosity'] = poros2
                    self.statistics['diameters']['ellipsoid_x'].append(diams_list[0])
                    self.statistics['diameters']['ellipsoid_y'].append(diams_list[1])
                    self.statistics['diameters']['ellipsoid_z'].append(diams_list[2])
                    self.statistics['diameters']['Total'].append(diams_list[0])
                    if angle_tolerance_flag == True and angles_range_flag == False:
                        self.statistics['angles']['theta_x'].append(angles[0])
                        self.statistics['angles']['theta_y'].append(angles[1])
                        self.statistics['angles']['theta_z'].append(angles[2])
                    """ A counter of the number of balls generated"""
                    n_balls += 1
                    
                if n_balls % 100 == 0:
                    print(" Ellipsoids created: ", n_balls)
                    print(f" Porosity: {self.statistics['porosity']:.4f}")
                    print(f" Ellipsoid of size: {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f} inserted.")
                    print("")
                # if n_balls >= counts[0]*1.5 or self.statistics['porosity'] <= 1.05*target_porosity:
                # if n_balls >= 10:
                # if self.statistics['porosity'] <= 1.025*target_porosity:
                #     break     
    
        self.im[self.im>1] -= 1
        self.im.astype(np.uint16)
    
    def add_one_sphere(self, diameter: int,
                       mode: int = 0):
        """
        Adds 1 sphere to the structure using the Random Sequential Addition (RSA).
    
        Parameters:
        - target_porosity: Desired porosity level for the final structure.
        - diameter: Diameter value.
        - cfd: Cumulative frequency distribution for the sphere diameters.
        - mode: If set, enables random size selection based on ratios (used for 2 types of AM).
        """
        # Check if the structure's image is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
    
        rad = int(diameter * 0.5 / self.voxel + 0.5)  # Calculate the radius in voxels.

        # Generate a new sphere using the RSA (Random Sequential Addition) method.
        self.im = 1*ps.generators.rsa(im_or_shape=self.im, r=rad, n_max=1)
        
        # Store the diameter of the newly added sphere.
        self.statistics['diameters']['Total'].append(diameter)
        # Ensure that all sphere values are set to 1 in the final image.
        self.im[self.im >= 1] = 1
        # indices = np.argwhere(self.im == 1)
        # Step 2: Find the bounding box of the sphere
        # margin = 5 
        # upper_margin = 5
        # lower_margin = 5
        # self.upper_margin = upper_margin
        # self.lower_margin = lower_margin
        # lowest_index = indices[:, 2].min()   # Min Z-axis (depth)
        # uppermost_index = indices[:, 2].max()  # Max Z-axis (depth)
        # leftmost_index = indices[:, 1].min()  # Min Y-axis (horizontal)
        # rightmost_index = indices[:, 1].max()  # Max Y-axis (horizontal)
        # frontmost_index = indices[:, 0].min()  # Min X-axis (vertical)
        # backmost_index = indices[:, 0].max()  # Max X-axis (vertical)
        # # Step 3: Crop the array tightly around the detected limits      
        # self.im = self.im[frontmost_index:backmost_index+1, leftmost_index:rightmost_index+1, lowest_index:uppermost_index+1]       
        # # Step 4: Add a 5-voxel boundary (padding) to the cropped array
        # padded_shape = (self.im.shape[0] + 2 * margin, self.im.shape[1] + 2 * margin, self.im.shape[2] + lower_margin + upper_margin)        
        # # Create a new array filled with 0s
        # padded_array = np.zeros(padded_shape, dtype=self.im.dtype)       
        # # Step 5: Place the cropped array inside the center of the padded array
        # padded_array[margin:margin+self.im.shape[0], margin:margin+self.im.shape[1], margin:margin+self.im.shape[2]] = self.im
        # self.im = padded_array
        self.im.astype(np.uint16)  # Convert image to 16-bit unsigned integer type.

    def add_one_ellipsoid(self, diams: npt.NDArray[np.float64],
                          cfd: npt.NDArray[np.float64], 
                          mode: int = 0,
                          electrode_generation_mode: str = 'extended',
                          theta_x: int = 0,
                          theta_y: int = 0,
                          theta_z: int = 0):

        # Check if the structure's image is properly defined.
        if self.im is None:
            raise ValueError('You did not define a box or a voxel size for your structure')
        
        diams_list = [[], [], []]
        self.statistics['diameters']['x'] = []
        self.statistics['diameters']['y'] = []
        self.statistics['diameters']['z'] = []
        self.statistics['angles']['theta_x'] = []
        self.statistics['angles']['theta_y'] = []
        self.statistics['angles']['theta_z'] = []
        
        # Randomly select a sphere diameter based on the cumulative frequency distribution.
        rnd = np.random.random_sample()
        idmin = (np.abs(cfd[mode] - rnd)).argmin()  # Find the nearest diameter index.
        diam = diams[idmin]  # Select the corresponding diameter.
        
        factor_y = random.uniform(0.5, 0.75)
        factor_z = random.uniform(0.15, 0.4)
        diams_x = diam
        diams_y = factor_y * diams_x  # Calculate y-diameter based on factor.
        diams_z = factor_z * diams_x  # Calculate z-diameter based on factor.        

        # Calculate radii in voxel units.
        radii = [diams_x * 0.5 / self.voxel,
                  diams_y * 0.5 / self.voxel,
                  diams_z * 0.5 / self.voxel]
        diams_list[0] = diams_x
        diams_list[1] = diams_y
        diams_list[2] = diams_z

        self.im, angles = _make_ellipsoid(round(radii[0]), round(radii[1]), round(radii[2]), smooth=True, rotation_angles=True, theta_x=np.radians(theta_x),
                                          theta_y=np.radians(theta_y), theta_z=np.radians(theta_y))
                
        self.statistics['diameters']['x'].append(diams_x)
        self.statistics['diameters']['y'].append(diams_y)
        self.statistics['diameters']['z'].append(diams_z)
        self.statistics['diameters']['Total'].append(diams_x)
        self.statistics['angles']['theta_x'].append(np.radians(theta_x))
        self.statistics['angles']['theta_y'].append(np.radians(theta_y))
        self.statistics['angles']['theta_z'].append(np.radians(theta_z))       
        
        # Ensure that all sphere values are set to 1 in the final image.
        self.im[self.im >= 1] = 1
        # indices = np.argwhere(self.im == 1)
        # # Step 2: Find the bounding box of the sphere
        # margin = 5 
        # upper_margin = 5
        # lower_margin = 5
        # lowest_index = indices[:, 2].min()   # Min Z-axis (depth)
        # uppermost_index = indices[:, 2].max()  # Max Z-axis (depth)
        # leftmost_index = indices[:, 1].min()  # Min Y-axis (horizontal)
        # rightmost_index = indices[:, 1].max()  # Max Y-axis (horizontal)
        # frontmost_index = indices[:, 0].min()  # Min X-axis (vertical)
        # backmost_index = indices[:, 0].max()  # Max X-axis (vertical)
        # # Step 3: Crop the array tightly around the detected limits      
        # self.im = self.im[frontmost_index:backmost_index+1, leftmost_index:rightmost_index+1, lowest_index:uppermost_index+1]       
        # # Step 4: Add a 5-voxel boundary (padding) to the cropped array
        # padded_shape = (self.im.shape[0] + 2 * margin, self.im.shape[1] + 2 * margin, self.im.shape[2] + lower_margin + upper_margin)        
        # # Create a new array filled with 0s
        # padded_array = np.zeros(padded_shape, dtype=self.im.dtype)       
        # # Step 5: Place the cropped array inside the center of the padded array
        # padded_array[margin:margin+self.im.shape[0], margin:margin+self.im.shape[1], margin:margin+self.im.shape[2]] = self.im
        # self.im = padded_array
        self.im.astype(np.uint16)  # Convert image to 16-bit unsigned integer type.
                
        
    def clear_binder(self):
        """
        Clears the binder phase from the image.
        Any voxel with a value of 2 (indicating the binder) is set to 0.
        The image array is then cast to an unsigned 16-bit integer type.
        """
        self.im[self.im == 2] = 0  # Set all voxels labeled as binder (value 2) to 0.
        self.im.astype(np.uint16)  # Convert the image data type to unsigned 16-bit integers.
    
    def add_binder_random(self, vol_frac: float, weight: float, pre_fraction: float, multiplier: int):
        """
        Randomly adds a binder phase to the image based on specified volume fraction and weight.
        
        Args:
            vol_frac: Target volume fraction of the binder phase.
            weight: Probability weight for determining binder addition strategy.
            pre_fraction: Fraction of the total volume to pre-fill before random filling.
            multiplier: Factor to control how many additional voxels are filled based on the initial fill.
    
        Raises:
            ValueError: If weight is outside the range [0.0, 1.0].
        """
        # start_time = time.time()  # Record start time for operation duration tracking.
        
        # Pad the image to handle edge cases without running out of bounds.
        padarray = np.pad(self.im, ((1, 1), (1, 1), (1, 1)), 'symmetric', reflect_type='even')
        
        # Calculate the total number of voxels that need to be filled based on the target volume fraction.
        num_vox = self.calculate_voxel_num(padarray, vol_frac=vol_frac)
    
        # Validate the weight to ensure it's within the acceptable range.
        if weight < 0.0 or weight > 1.0:
            raise ValueError('Weight has to be in the range [0.0, 1.0], retry with another value')
    
        # Determine how many voxels to pre-fill with the binder.
        num_pre = int(num_vox * pre_fraction)
        
        # Get the coordinates of all voxels that are currently empty (value 0).
        ip, jp, kp = np.nonzero(padarray == 0)
        
        # Randomly select indices to pre-fill with the binder.
        ipr = np.random.choice(len(ip), num_pre, replace=False)
        padarray[ip[ipr], jp[ipr], kp[ipr]] = 2  # Set selected indices to binder (value 2).
    
        cumulative_vol = num_pre  # Initialize cumulative volume with the pre-filled amount.
        
        # Continue adding binder until the target volume fraction is reached.
        while num_vox > cumulative_vol:
            # Check if the operation is taking too long and stop if it exceeds 30 seconds.
            # if time.time() - start_time > 30:
            #     print("Time exceeded.")
            #     print("Time: ", time.time()-start_time)
            #     break
                
            val = np.random.random_sample()  # Generate a random number between 0 and 1.
            
            if val < weight:
                # If the random value is less than the weight, add to the binder border voxels.
                cbd_vox = get_some_borders(padarray, 0, 2)  # Get border voxels adjacent to the binder phase.
                
                # If we can fill more voxels than available borders, fill all.
                if num_pre * multiplier >= cbd_vox[0].shape[0]:
                    padarray[tuple(cbd_vox)] = 2  # Set border voxels to binder.
                    cumulative_vol += cbd_vox[0].shape[0]  # Update cumulative volume.
                else:
                    # Randomly select some border voxels to fill.
                    idcbd = np.random.choice(np.arange(cbd_vox[0].shape[0]), num_pre * multiplier)
                    padarray[tuple(idx[idcbd] for idx in cbd_vox)] = 2  # Set selected voxels to binder.
                    cumulative_vol += num_pre * multiplier  # Update cumulative volume.
            else:
                # Otherwise, add binder to the neighboring voxels of both phases 1 and 3.
                am1_vox = get_some_borders(padarray, 0, 1)  # Get border voxels adjacent to phase 1.
                am2_vox = get_some_borders(padarray, 0, 3)  # Get border voxels adjacent to phase 3.
    
                # Combine the border voxel coordinates.
                am_vox = [np.append(am1, am2) for am1, am2 in zip(am1_vox, am2_vox)]
                
                # If we can fill more voxels than available borders, fill all.
                if num_pre * multiplier >= am_vox[0].shape[0]:
                    padarray[tuple(am_vox)] = 2  # Set neighboring voxels to binder.
                    cumulative_vol += am_vox[0].shape[0]  # Update cumulative volume.
                else:
                    # Randomly select some neighboring voxels to fill.
                    idam = np.random.choice(np.arange(am_vox[0].shape[0]), num_pre * multiplier)
                    padarray[tuple(idx[idam] for idx in am_vox)] = 2  # Set selected voxels to binder.
                    cumulative_vol += num_pre * multiplier  # Update cumulative volume.
    
        # Update the original image with the modified padded array, removing the padding.
        self.im = padarray[1:-1, 1:-1, 1:-1]
    
    def dilate(self, phase: int, into: int, shells: int):
        """
        Dilates a specified phase within the image by a certain number of shells.
        
        Args:
            phase: The phase to dilate (represented by an integer label).
            into: The phase that the dilation occurs into (also represented by an integer label).
            shells: The number of shells (layers of voxels) to dilate.
        """
        # Pad the image to prevent edge effects during dilation.
        padarray = np.pad(self.im, ((1, 1), (1, 1), (1, 1)), 'symmetric', reflect_type='even')
        
        # Perform dilation over the specified number of shells.
        for i in range(shells):
            # Get the coordinates of border voxels adjacent to the specified phase.
            voxels = get_some_borders(padarray, into, phase)
            padarray[voxels] = phase  # Set the border voxels to the specified phase.
    
        # Update the original image with the modified padded array, removing the padding.
        self.im = padarray[1:-1, 1:-1, 1:-1]
    
    def erode(self, phase: int, inrelationto: int, target_vol_frac: float):
        """
        Erodes a specified phase in the image until a target volume fraction is reached.
        
        Args:
            phase: The phase to erode (represented by an integer label).
            inrelationto: The phase to which the erosion is in relation.
            target_vol_frac: The target volume fraction for the specified phase.
        """
        # Pad the image to handle boundary conditions during erosion.
        padarray = np.pad(self.im, ((1, 1), (1, 1), (1, 1)), 'symmetric', reflect_type='even')
    
        # Calculate the target number of voxels based on the target volume fraction.
        target_voxels = self.calculate_voxel_num(padarray, vol_frac=target_vol_frac)
        neight = get_all_borders(padarray, phase)  # Get border voxels of the specified phase.
        boundary_voxels = neight[0].shape[0]  # Count how many boundary voxels exist.
        
        # Determine how many voxels need to be deleted to reach the target volume.
        voxels_to_delete = np.count_nonzero(padarray == phase) - target_voxels
    
        # Loop to erode the phase until the desired volume fraction is reached.
        while voxels_to_delete >= boundary_voxels:
            voxels = get_some_borders(padarray, phase, inrelationto)  # Get border voxels for deletion.
            padarray[voxels] = 0  # Set the selected voxels to 0 (erode).
            
            neight = get_all_borders(padarray, phase)  # Re-evaluate border voxels after deletion.
            boundary_voxels = neight[0].shape[0]  # Update boundary voxel count.
            
            voxels_to_delete_a = voxels_to_delete  # Keep track of voxels to delete.
            voxels_to_delete = np.count_nonzero(padarray == phase) - target_voxels  # Update remaining voxels to delete.
            
            # Break the loop if no further voxels can be deleted.
            if voxels_to_delete == voxels_to_delete_a:
                break
                
            print(" Voxels to delete 1st loop: ", voxels_to_delete)  # Debug output for tracking deletion count.
    
        # Second loop to erode phase until the target volume fraction is achieved.
        while voxels_to_delete > 0:
            voxels = get_some_borders(padarray, phase, inrelationto)  # Get border voxels for deletion.
            voxels_check = voxels[0].tolist()  # Convert voxel coordinates to a list for checking.
            
            # Handle cases where the voxel array might be empty to avoid errors.
            if not voxels_check:  # Check if there are no border voxels left to delete.
                break
                
            choice = np.random.choice(len(voxels[0]), voxels_to_delete)  # Randomly select which voxels to delete.
            padarray[tuple(idx[choice] for idx in voxels)] = 0  # Set the selected voxels to 0 (erode).
            
            # Update the count of remaining voxels to delete.
            voxels_to_delete = np.count_nonzero(padarray == phase) - target_voxels
            
            # Uncomment for debugging to track deletion count.
            # print("Voxels to delete 2nd loop: ", voxels_to_delete)        
    
        # Update the original image with the modified padded array, removing the padding.
        self.im = padarray[1:-1, 1:-1, 1:-1]
    
    def add_binder_mistry(self, vol_frac: float, weight: float, pre_fraction: float, multiplier: int,
                          surface_nucleate: bool = False, binder_label: int = 2):
        """
        Adds a binder phase to the image using the Mistry method, which allows for controlled binder distribution.
        
        Args:
            vol_frac: Target volume fraction of the binder phase.
            weight: Probability weight for determining binder addition strategy.
            pre_fraction: Fraction of the total volume to pre-fill before random filling.
            multiplier: Factor to control how many additional voxels are filled based on the initial fill.
            surface_nucleate: If True, nucleate binder only at surface voxels.
            binder_label: The integer label to use for the binder phase.
        """
        # Pad the image to handle edge effects during addition.
        padarray = np.pad(self.im, ((1, 1), (1, 1), (1, 1)), 'symmetric', reflect_type='even')
        
        # Calculate the total number of voxels to be filled based on the target volume fraction.
        num_vox = self.calculate_voxel_num(padarray, vol_frac=vol_frac)
        
        # Validate the weight to ensure it's within the acceptable range.
        if weight < 0.0 or weight > 1.0:
            raise ValueError('Weight has to be in the range [0.0, 1.0], retry with another value')
    
        # Determine how many voxels to pre-fill with the binder.
        num_pre = int(num_vox * pre_fraction)
    
        # Get the coordinates of all voxels that are currently empty (value 0).
        ip, jp, kp = np.nonzero(padarray == 0)
        
        # If surface_nucleate is enabled, only consider surface voxels for pre-filling.
        if surface_nucleate:
            ip, jp, kp = get_all_borders(padarray, 0)  # Get border voxels of the empty phase.
    
        # Randomly select indices to pre-fill with the binder.
        ipr = np.random.choice(len(ip), num_pre, replace=False)
        padarray[ip[ipr], jp[ipr], kp[ipr]] = binder_label  # Set selected indices to binder label.
    
        start_time = time.time()  # Record the start time for the addition process.
        
        # Call the loop_mistry function to add binder with specified parameters.
        self.im = loop_mistry(padarray, num_vox, num_pre, num_pre * multiplier, weight, binder_label, 0)[1:-1, 1:-1, 1:-1]
        
        print("CBD time: ", time.time()-start_time)  # Output the time taken for the binder addition.
    
    def add_binder_blob(self, vol_frac: float, increment: float, binder_label: int = 2):
        """
        Adds a binder phase to the image by expanding blobs based on local thickness.
        
        Args:
            vol_frac: Target volume fraction of the binder phase.
            increment: Step size for increasing the thickness of added binder.
            binder_label: The integer label to use for the binder phase.
        """
        start_time = time.time()  # Record the start time for the addition process.
        
        # Calculate the total number of voxels to be filled based on the target volume fraction.
        num_vox = self.calculate_voxel_num(self.im, vol_frac=vol_frac)
        
        mask = self.im == 0  # Create a mask for empty voxels.
        
        # Calculate the local thickness of the empty voxels to guide binder addition.
        local_thickness = ps.filters.local_thickness(mask)
        
        cumulative_vol = 0  # Initialize the cumulative volume of added binder.
        d = 0.0  # Initialize the thickness increment.
    
        # Loop to add binder until the desired volume is reached.
        while num_vox > cumulative_vol:
            if time.time() - start_time > 30:  # Stop if the process exceeds 30 seconds.
                print("Time exceeded.")
                print("Time: ", time.time()-start_time)
                break
                
            # Create a mask for regions suitable for binder addition based on local thickness.
            mask_add = (local_thickness > 0.00000001) & (local_thickness < d)
            cumulative_vol = np.count_nonzero(mask_add)  # Count the number of voxels available for addition.
            self.im[mask_add] = binder_label  # Set suitable voxels to binder label.
            d += increment  # Increase the thickness increment for the next iteration.
    
        # Calculate how many excess voxels were added beyond the target volume.
        rest = cumulative_vol - num_vox
    
        # Create a new mask for the added binder phase.
        mask = self.im == binder_label
        local_thickness = ps.filters.local_thickness(mask)  # Calculate local thickness of the binder phase.
        
        cumulative_vol = 0  # Reset cumulative volume for removal.
        d = 0.0  # Reset thickness increment.
    
        # Loop to remove binder until the target volume is achieved.
        while rest > cumulative_vol:
            # Create a mask for regions suitable for binder removal based on local thickness.
            mask_add = (local_thickness > 0.00000001) & (local_thickness < d)
            cumulative_vol = np.count_nonzero(mask_add)  # Count the number of voxels available for removal.
            self.im[mask_add] = 0  # Set suitable voxels back to 0 (remove binder).
            d += increment  # Increase the thickness increment for the next iteration.
    
    def add_binder_bridge(self, vol_frac: float, increment: float, binder_label: int = 2):
        """
        Adds a binder phase to the image by expanding bridges based on local thickness.
        
        Args:
            vol_frac: Target volume fraction of the binder phase.
            increment: Step size for increasing the thickness of added binder.
            binder_label: The integer label to use for the binder phase.
        """
        
        # Calculate the total number of voxels to be filled based on the target volume fraction.
        num_vox = self.calculate_voxel_num(self.im, vol_frac=vol_frac)
        
        # Calculate the local thickness of the empty voxels to guide binder addition.
        local_thickness = ps.filters.local_thickness(self.im == 0)
        
        cumulative_vol = 0  # Initialize the cumulative volume of added binder.
        d = 0.0  # Initialize the thickness increment.
    
        # Loop to add binder until the desired volume is reached.
        while num_vox > cumulative_vol:
            # Create a mask for regions suitable for binder addition based on local thickness.
            mask_add = (local_thickness > 0.00000001) & (local_thickness < d)
            cumulative_vol = np.count_nonzero(mask_add)  # Count the number of voxels available for addition.
            self.im[mask_add] = binder_label  # Set suitable voxels to binder label.
            d += increment  # Increase the thickness increment for the next iteration.

    
class VoronoiStructure(Structure):

    def __init__(self, box: List[float] = None, voxel: float = None):
        """ box: A list containing the size of the simulation box in micrometers
            voxel: A float representing the size of each voxel in micrometers

            This class generates a polycrystaline model based on a Voronoi partition
             0: Pore, 1 - N: Voronoi domain """

        super().__init__(box=box, voxel=voxel)

        self.statistics = dict()
        self.statistics['diameters'] = []
        self.statistics['volumes'] = []
        self.statistics['porosity'] = 1.0

    def build_tesselation(self, n: int):
        """rho: density of points, defines """
        self.im = self.voronoi_partition(box=self.box, voxel=self.voxel, n=n)

    def get_statistics(self):
        unique, counts = np.unique(self.im, return_counts=True)

        uqvals = np.asarray((unique, counts)).T
        uqvals = uqvals[uqvals[:, 0].argsort()]

        for cell in uqvals:
            volume = cell[1] * self.voxel ** 3
            diameter = (6.0 * volume / np.pi) ** (1.0 / 3.0)
            self.statistics['volumes'] = cell[1] * self.voxel ** 3
            self.statistics['diameters'] = diameter
            self.statistics['porosity'] = self.calculate_porosity(array=self.im)

    @staticmethod
    @njit
    def voronoi_partition(box: npt.NDArray[float], voxel: float, n: int):

        x = box[0] * np.random.random_sample((n,))
        y = box[1] * np.random.random_sample((n,))
        z = box[2] * np.random.random_sample((n,))

        points = np.column_stack((x, y, z))

        # Define mesh
        x_ = np.linspace(-box[0] / 2.0, box[0] / 2.0, int(box[0] / voxel))
        y_ = np.linspace(-box[1] / 2.0, box[1] / 2.0, int(box[1] / voxel))
        z_ = np.linspace(-box[2] / 2.0, box[2] / 2.0, int(box[2] / voxel))

        M = np.zeros((x_.size, y_.size, z_.size), dtype=np.int16)
        M.fill(-1)

        for i in range(x_.size):
            for j in range(y_.size):
                for k in range(z_.size):
                    coord = np.array([i, j, k]).astype(np.float64) * voxel
                    diff = points - coord
                    dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)
                    closest = np.argmin(dist)
                    M[i, j, k] = closest

        return M + 1

####################################################################################################################
#                                       DEFINITIONS OF NUMBA FUNCTIONS                                             #
####################################################################################################################
@njit
def calc_centroid(arr):

    #arr: A 3D ndarray of dtype=int
    #label: A integer value representing the value of the particle of interest,
    #n: Number of voxels with label == label

    dims = np.shape(arr)

    m000 = 0
    m100 = 0
    m010 = 0
    m001 = 0
    for i in np.arange(dims[0]):
        for j in np.arange(dims[1]):
            for k in np.arange(dims[2]):
                m000 = m000 + arr[i,j,k]
                m100 = m100 + arr[i,j,k]*i
                m010 = m010 + arr[i,j,k]*j
                m001 = m001 + arr[i,j,k]*k

    m000 = m000 + 0.0000001
    m100 = m100 + 0.0000001
    m010 = m010 + 0.0000001
    m001 = m001 + 0.0000001

    return [m100/m000,m010/m000,m001/m000]

@stencil
def borders_all_stencil(x: npt.NDArray[np.uint16]):
    condition =\
        (not x[1, 0, 0] or not x[-1, 0, 0]
         or not x[0, 1, 0] or not x[0, -1, 0]
         or not x[0, 0, 1] or not x[0, 0, -1]) and x[0, 0, 0]

    return condition


@stencil
def borders_some_stencil(x: npt.NDArray[np.uint16], domain: int, relative: int):
    condition =\
        (x[1, 0, 0] == relative or x[-1, 0, 0] == relative or
         x[0, 1, 0] == relative or x[0, -1, 0] == relative or
         x[0, 0, 1] == relative or x[0, 0, -1] == relative) and x[0, 0, 0] == domain

    return condition


@njit
def get_all_borders(array: npt.NDArray[np.uint16], domain: int):
    mask = array == domain
    bordermask = borders_all_stencil(mask)

    return np.nonzero(bordermask)


@njit
def get_some_borders(array: npt.NDArray[np.uint16], domain: int, relative: int):
    bordermask = borders_some_stencil(array, domain, relative)
    return np.nonzero(bordermask)


@njit
def deposition_probability(
        array: npt.NDArray[np.uint16], siteidx: npt.NDArray[np.uint16], weight: float, domain: int, into: int):

    stencil6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int16)

    probabilities = np.zeros(siteidx.shape[0])
    cumulative_prob = 0.0
    for site in np.arange(siteidx.shape[0]):
        num_am = 0
        num_add = 0
        for d in np.arange(stencil6.shape[0]):
            sample_index = siteidx[site, :] + stencil6[d, :]
            i = sample_index[0]
            j = sample_index[1]
            k = sample_index[2]

            condition = (array.shape[0] > i >= 0) and \
                        (array.shape[1] > j >= 0) and \
                        (array.shape[2] > k >= 0)

            if condition:
                if array[i, j, k] != into and array[i, j, k] != domain:
                    num_am += 1
                elif array[i, j, k] == domain:
                    num_add += 1

        cumulative_prob += ((1.0 - weight) * num_am + weight * num_add) / 6.0
        probabilities[site] = cumulative_prob

    return probabilities


@njit
def deposition_sites(array: npt.NDArray[np.uint16], interphase_sites: npt.NDArray[np.uint16],
                     probabilities: npt.NDArray[np.float64], num: int, domain: int):

    for _ in np.arange(num):
        val = np.random.random_sample() * probabilities[-1]
        dist = np.abs(probabilities - val)
        row_i = np.argmin(dist)

        i = interphase_sites[row_i, 0]
        j = interphase_sites[row_i, 1]
        k = interphase_sites[row_i, 2]

        array[i, j, k] = domain

    return array


@njit
def loop_mistry(array: npt.NDArray[np.uint16], maxnum: int, cumulative: int, num: int, weight: float, domain: int, into: int):
    
    while maxnum > cumulative:

        neigh = get_all_borders(array, into)
        siteidx = np.column_stack((neigh[0], neigh[1], neigh[2]))
        probability = deposition_probability(array, siteidx, weight, domain, into)
        array = deposition_sites(array, siteidx, probability, num, domain)
        cumulative += num

    return array
