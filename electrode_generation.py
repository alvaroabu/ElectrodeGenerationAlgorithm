from electrode_generator_auto import ElectrodeGeneratorAuto
from itertools import product

""" Conversions """
cm3toum3 = 1e+12
cm2toum2 = 1e+8

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

total_structures = len(thickness)*len(porosity)

parameter_set = parameter_set_init
test_set = test_set_init

for s, tries, vox, CBD_nano, thick, over, CBD_meth, (m0, m1, m2) in product(
        size, max_tries_list, voxel, CBD_nanoporosity, thickness, overlap_list, cbd_method, mass):
    
    test_set += 1
    total_structures = len(thickness) * len(porosity)

    for c, poro in enumerate(porosity, start=1):
        Electrode = ElectrodeGeneratorAuto(
            cm3toum3=cm3toum3,
            target_mass_form_am=m0, 
            target_mass_form_binder=m1,
            target_mass_form_carbon=m2, 
            dens_am=rho_graphite,
            dens_binder=rho_binder,
            dens_carbon=rho_carbon, 
            CBD_nanoporosity=CBD_nano, 
            target_porosity=poro,
            size=s,
            thickness=thick, 
            sph_method=sph_method, 
            cbd_method=CBD_meth, 
            cap_am=cap_Gr,
            actual_struct=c, 
            total_struct=total_structures, 
            overlap=over,
            max_tries=tries, 
            cut=cut, 
            variable_vox_size=variable_vox_size,
            parameter_set=parameter_set, 
            test=test_set, 
            geometry=geometry, 
            electrode_generation_mode=electrode_generation_mode, 
            cm2toum2=cm2toum2,
            ind_mod=ind_mod, 
            counter=c, 
            voxel=vox, 
            filename_extra=f"OVLP{over}-CBD_{CBD_meth[:3]}-nP{CBD_nano}-PS{parameter_set}-TS{test_set}_Vox{vox}",
            sph_down=sph_down,
            channel=channel,
            angle_tolerance_flag=angle_tolerance_flag,
            angle_tolerance=angle_tolerance, 
            angles_range_flag=angles_range_flag,
            theta_x=theta_x, 
            theta_y=theta_y,
            theta_z=theta_z,
            material='Graphite'

        )
