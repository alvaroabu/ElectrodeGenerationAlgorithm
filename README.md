# Electrode Generator Algorithm

A Python framework for the **stochastic generation of voxelized 3D electrode microstructures** for battery simulations.  
It combines **particle packing**, **binder/carbon domain (CBD) addition**, and **statistical validation** into an automated workflow.

---

## Features
- **Automatic electrode generation**
  - Particle geometries: sphere, ellipsoid, or mixed.
  - Packing methods: Random Sequential Addition (**RSA**) or Gravity packing.
- **Volume fraction & porosity control**
  - Target porosity and nanoporosity correction for CBD.
- **Particle size distribution validation**
  - Ensures PSD follows reference SEM-derived lognormal distributions.
- **CBD addition algorithms**
  - `bridge`, `mistry`, `random`, `blob`.
- **Flexible exports**
  - Save microstructures as **HDF5**, **VTK**, or **Pickle**.
- **Optional extras**
  - Padding & cropping.
  - Channel carving along X-axis.
  - Ellipsoid orientation with configurable rotation angles.

---

## Workflow
1. **Define inputs**
   - Capacity of active material (AM).
   - Mass ratios of AM, binder, and carbon.
   - Material densities, porosity, and CBD nanoporosity.
   - Electrode size and thickness.
2. **Generate particles**
   - Choose particle geometry: sphere, ellipsoid, or mixed.
   - Select packing method: RSA or gravity-based packing.
3. **Validate PSD**
   - Compare generated PSD against SEM-derived lognormal distribution.
   - Structures labeled as *Good* or *Bad*.
4. **Add CBD**
   - Insert binder/carbon domain with the chosen method.
5. **Export results**
   - Structure statistics, effective porosity, mass loading, and total capacity.
   - Export 3D voxelized electrode as `.vtk` or `.h5`.

---

## Example Parameters
```python
""" Global parameters """
""" Reference capacity """
cap_Gr = 372.00 # mAh / g // Theoretical value
""" Density inputs """
rho_graphite = 2.2363/cm3toum3 #g/cm^3 graphite (From LG basic)
rho_carbon = 1.85/cm3toum3 #g/cm^3 C65
rho_binder = 1.6045/cm3toum3 #g/cm^3 CMC
 
""" Variable parameters """
mass = [[90, 5, 5], [92, 4, 4], [94, 3, 3], [96, 2, 2]]

thickness = [75]
porosity = [0.35]
CBD_nanoporosity = [0.476] # From M. Chouchane, 2021
size = [50] # Only square electrodes are considered

geometry = 'sphere' # 'half-sp-half-el', 'sphere', 'ellipsoid'
sph_method = 'rsa_CFD' # Sphere generation method (rsa_CFD, gravity_CFD)
electrode_generation_mode = 'extended' # (periodic, extended, contained)
cbd_method = ['bridge'] # CBD addition method (bridge, random, mistry, blob)
voxel = [1] # 0.25, 0.5, 1, 1.5, 2

""" Test and parameter set numbers """
parameter_set_init = 1
test_set_init = 0
""" Max tries in ball generation (only for gravity_CFD) """
max_tries_list = [1000]

""" Overlap factor """
overlap_list = [0.1]

""" Config. options """
""" Cut or not the electrode (only for gravity method) """
cut = False
""" Variable voxel size """
variable_vox_size = False
""" Individual labelling for particles """
ind_mod = True
""" Spheres / ellipsoids up / down """
sph_down = False
""" Tunnel """
channel = False

""" Rotation angles configuration """
angle_tolerance_flag = False
angle_tolerance = 20 
angles_range_flag = False
theta_x = 0 
theta_y = 0
theta_z= 0
```

---

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib
- h5py
- VTK
- PoresPy → A modified PoresPy fork is required (with custom ellipsoid functions).
