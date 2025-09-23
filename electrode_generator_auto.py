import itertools  
import os
import pickle
import random
import time
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from calculator_v2 import FormulationDry
from signals import lognormal, cfd_lognormal
from structure import StructureSpheres


class ElectrodeGeneratorAuto:
    """High‑level orchestrator to generate voxelized electrode microstructures.

    Steps:
      1) Validate inputs and set up output directories.
      2) Compute target volume fractions from mass formulation and densities.
      3) Generate particles (sphere/ellipsoid/mixed) via RSA/Gravity + target CFD.
      4) Check particle‑size distribution vs reference lognormal (good/bad structure gate).
      5) Add CBD (binder/carbon domain) and erode to target volume.
      6) Compute output stats, capacities, and export VTK/HDF5/PKL.
      7) Optionally carve channels along X.

    """

    def __init__(
        self,
        cm3toum3: float,
        dens_am: Union[float, List[Union[Tuple[float], float]]],
        dens_binder: float,
        dens_carbon: float,
        size: int,
        thickness: int,
        sph_method: str,
        cap_am: Union[float, List[Union[Tuple[float], float]]],
        actual_struct: int,
        parameter_set: str,
        test: str,
        geometry: str,
        cm2toum2: float,
        counter: int,
        voxel: float = 1.0,
        target_mass_form_am: int = 0,
        target_mass_form_binder: int = 0,
        target_mass_form_carbon: int = 0,
        CBD_nanoporosity: float = 0.0,
        target_porosity: float = 0,
        cbd_method: str = 'bridge',
        total_struct: int = 0,
        overlap: float = 0.0,
        max_tries: int = int(1e4),
        electrode_generation_mode: str = '',
        variable_vox_size: bool = False,
        ind_mod: bool = False,
        cut: bool = False,
        filename_extra: str = "",
        vol_distrib: bool = True,
        sph_down: bool = True,  
        channel: bool = True,
        wall_th: List[int] = [40, 40, 40, 30, 30, 30, 15, 15, 15], 
        channel_th: List[int] = [15, 30, 40, 15, 30, 40, 15, 30, 40],  
        _1_particle: bool = False,
        diameter: int = 10,
        margins: List[int] = [5, 5, 5, 5],
        angle_tolerance_flag: bool = True,
        angle_tolerance: int = 10,
        angles_range_flag: bool = True,
        theta_x: int = 0,
        theta_y: int = 0,
        theta_z: int = 0,
        material: str = 'material',
    ):
        """Validate inputs, create output directories, compute targets, generate structure.

        Unit expectations: size/thickness/margins/diameter in µm; voxel in µm/voxel.
        Densities in g/µm³ if you later multiply by a µm³ volume.
        """

        # Abbreviated material tag for filenames (keep SC83 literal)
        mat_short = material if material == 'SC83' else material[:2]

        # Output base path
        if _1_particle:
            aux_path = (
                "./{material}Structures/" + sph_method + "/1_particle/ParameterSet" + str(parameter_set) + "/Test" + str(test) + "/"
            )
            if not os.path.exists(aux_path):
                os.makedirs(aux_path)
        else:
            mass_str = f"{target_mass_form_am}-{target_mass_form_binder}-{target_mass_form_carbon}"
            aux_path = (
                f"./{material}Structures/" + sph_method + "/" + mass_str + "/" + f"{size}x{size}um/" + "ParameterSet" + str(parameter_set) + "/Test" + str(test) + "/"
            )
            if not os.path.exists(aux_path):
                os.makedirs(aux_path)

            # Validate and print header
            self.exceptions(
                cbd_method,
                geometry,
                target_mass_form_am,
                target_mass_form_binder,
                target_mass_form_carbon,
                aux_path,
                counter,
            )
            self.print_init_info(
                size,
                CBD_nanoporosity,
                target_mass_form_am,
                target_mass_form_binder,
                actual_struct,
                total_struct,
                target_mass_form_carbon,
                sph_method,
                overlap,
                max_tries,
                geometry,
                cbd_method,
                thickness,
                target_porosity,
            )

            start = time.time()

            # Compute target volume fractions from formulation/densities/box
            self.formula = FormulationDry(
                mass_am=target_mass_form_am,
                mass_binder=target_mass_form_binder,
                mass_carbon=target_mass_form_carbon,
                dens_am=dens_am,
                dens_binder=dens_binder,
                dens_carbon=dens_carbon,
                CBD_nanoporosity=CBD_nanoporosity,
                box=[size, size, thickness],
                target_porosity=target_porosity,
            )

            vol_frac_am = self.formula.data['am']['vol_frac']
            vol_frac_CBD = self.formula.data['CBD']['vol_frac']

            # Pack input/calculated data into a dict
            input_data_list = [
                CBD_nanoporosity,
                target_porosity,
                cap_am,
                thickness,
                target_mass_form_am,
                target_mass_form_binder,
                target_mass_form_carbon,
            ]
            calculated_data_list = [vol_frac_am, vol_frac_CBD]
            self.data = self.input_calculated_data_2dict(input_data_list, calculated_data_list, material=mat_short)

        # Select reference lognormal parameters by material/geometry
        if material == 'Graphite':
            max_diam = 35
            if geometry == 'sphere':
                x0_orig, sigma_orig = 2.93, 0.33
            elif geometry == 'ellipsoid':
                x0_orig, sigma_orig = 2.94, 0.23
            elif geometry == 'half-sp-half-el':
                x0_orig, sigma_orig = [2.93, 2.94], [0.33, 0.23]
            else:
                raise ValueError(f"Geometry {geometry} not considered.")
        elif material == 'SC83':
            max_diam = 25
            if geometry == 'sphere':
                x0_orig, sigma_orig = 2.2925, 0.3914
            else:
                raise ValueError(f"Geometry {geometry} not considered.")
        else:
            raise ValueError(f"Material {material} not considered.")

        # Generate the (AM) structure
        self.generate_structure_add_AM(
            variable_vox_size,
            thickness,
            size,
            sph_method,
            overlap,
            max_tries,
            geometry,
            x0_orig,
            sigma_orig,
            electrode_generation_mode,
            ind_mod,
            voxel,
            vol_distrib=vol_distrib,
            _1_particle=_1_particle,
            diameter=diameter,
            margin=margins,
            angle_tolerance_flag=angle_tolerance_flag,
            angle_tolerance=angle_tolerance,
            angles_range_flag=angles_range_flag,
            theta_x=theta_x,
            theta_y=theta_y,
            theta_z=theta_z,
            mat_short=mat_short,
            max_diam=max_diam,
        )

        if _1_particle:
            self.pad_structure(margins, material=material)
            self.struct.to_hdf5(
                filename=aux_path
                + f"/{mat_short}-{geometry[:2].title()}-D{int(self.struct.statistics['diameters']['Total'][0])}-Vox{voxel}-LM{margins[0]}-UM{margins[1]}.h5"
            )
            self.struct.to_vtk(
                filename=aux_path
                + f"/{mat_short}-{geometry[:2].title()}-D{int(self.struct.statistics['diameters']['Total'][0])}-Vox{voxel}-LM{margins[0]}-UM{margins[1]}"
            )
        else:
            # Validate PSD against reference lognormal and make plots
            if geometry in ('sphere', 'ellipsoid'):
                diam = (
                    self.struct.statistics['diameters']['Total']
                    if geometry == 'sphere'
                    else self.struct.statistics['diameters']['x']
                )

                path, x0_factor, sigma_factor = self.check_diam_distribution_v2(diam, x0_orig, sigma_orig, geometry)

                psd_dir = aux_path + path + "ParticlesSizesDistribCheck/"
                if not os.path.exists(psd_dir):
                    os.makedirs(psd_dir)

                self.plot_particles_sizes_distrib_check(
                    psd_dir,
                    thickness,
                    x0_orig,
                    sigma_orig,
                    diam,
                    counter,
                    geometry=geometry
                )

            elif geometry == 'half-sp-half-el':
                # Validate sphere and ellipsoid subsets independently
                diam_el = self.struct.statistics['diameters']['ellipsoid_x']
                path_el, x0_factor_el, sigma_factor_el = self.check_diam_distribution_v2(
                    diam_el, x0_orig[1], sigma_orig[1], 'ellipsoid'
                )
                diam_sp = self.struct.statistics['diameters']['sphere']
                path_sp, x0_factor_sp, sigma_factor_sp = self.check_diam_distribution_v2(
                    diam_sp, x0_orig[0], sigma_orig[0], 'sphere'
                )

                path = path_el if path_el == path_sp else 'NoCBD/BadStructure/'

                psd_dir = aux_path + path + "ParticlesSizesDistribCheck/"
                if not os.path.exists(psd_dir):
                    os.makedirs(psd_dir)

                self.plot_particles_sizes_distrib_check(
                    psd_dir,
                    thickness,
                    x0_orig[0],
                    sigma_orig[0],
                    diam_sp,
                    counter,
                    geometry='sphere',
                )
                self.plot_particles_sizes_distrib_check(
                    psd_dir,
                    thickness,
                    x0_orig[1],
                    sigma_orig[1],
                    diam_el,
                    counter,
                    geometry='ellipsoid',
                )
            else:
                print("Geometry not considered.")

            if path == "NoCBD/BadStructure/":
                # Save minimal diagnostics for inspection
                with open(
                    aux_path
                    + path
                    + f"/{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}.pkl",
                    'wb',
                ) as f:
                    pickle.dump(self.struct.statistics['diameters']['Total'], f)
                self.struct.to_vtk(
                    aux_path
                    + path
                    + f"{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}"
                )
                self.struct.to_hdf5(
                    aux_path
                    + path
                    + f"{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}.h5"
                )

            if path == "NoCBD/GoodStructure/":
                path = path[:-1]  # Trim trailing slash

                # Optional cut for gravity packing
                if sph_method == 'gravity_CFD':
                    self.cutting(
                        path,
                        sph_method,
                        cut,
                        thickness,
                        size,
                        diam,
                        target_mass_form_am,
                        target_mass_form_binder,
                        target_mass_form_carbon,
                        dens_am,
                        dens_binder,
                        dens_carbon,
                        CBD_nanoporosity,
                        target_porosity,
                        aux_path,
                        self.axis,
                    )

                # Save structure stats (pre‑CBD)
                with open(
                    aux_path
                    + path
                    + f"/{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}.pkl",
                    'wb',
                ) as f:
                    pickle.dump(self.struct.statistics, f)

                self.struct.to_vtk(
                    aux_path
                    + path
                    + f"/{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}"
                )
                self.struct.to_hdf5(
                    aux_path
                    + path
                    + f"/{mat_short}-{geometry[:2].title()}-S{int(size)}-T{int(thickness)}-P{int(self.struct.statistics['porosity']*100)}-ID{counter}-{filename_extra}.h5"
                )

                # Add CBD and erode to target fraction
                self.CBD_addition_erosion(cbd_method, CBD_nanoporosity, vol_frac_CBD, ind_mod)
                self.duration = (time.time() - start) / 60  # minutes

                # Compute final stats
                vol_frac_list = [vol_frac_am, vol_frac_CBD, target_porosity]
                new_vol_frac_list = [
                    self.data['Calculated parameters'][f'{mat_short} volume fraction'],
                    self.data['Calculated parameters']['CBD volume fraction'],
                    self.data['Input parameters']['Target porosity'],
                ]
                vol = size * size * thickness  # total volume in µm³

                [
                    gen_str_am_vol_frac,
                    gen_str_cbd_vol_frac,
                    pore_vol_frac,
                    tot_vol_frac,
                ], [mass_am_in_box, totcap] = self.obtain_stats(
                    sph_method,
                    CBD_nanoporosity,
                    vol,
                    cap_am,
                    dens_am,
                    vol_frac_list,
                    new_vol_frac_list,
                    ind_mod,
                    mat_short=mat_short,
                )

                output_vol_frac_list = [gen_str_am_vol_frac, gen_str_cbd_vol_frac, pore_vol_frac]
                rho_list = [dens_am, self.formula.dens_CBD]
                m_list = [self.formula.data['am']['target_mass'], self.formula.data['CBD']['target_mass']]

                loading, self.data, self.structure_verification, path = self.output_data2dict(
                    output_vol_frac_list,
                    vol_frac_list,
                    new_vol_frac_list,
                    vol,
                    rho_list,
                    m_list,
                    [self.struct.box[0], self.struct.box[1]],
                    self.duration,
                    cm2toum2,
                    totcap,
                    sph_method,
                    aux_path,
                    self.data,
                    mat_short=mat_short,
                )

                # Final exports labeled with S/T/P/L/ID
                self.struct.to_hdf5(
                    filename=aux_path
                    + path
                    + f'/{mat_short}-{geometry[:2].title()}-S{int(self.struct.box[0])}-T{int(thickness)}-P{int(pore_vol_frac*100)}-L{loading:.2f}-ID{counter}-{filename_extra}.h5'
                )
                self.struct.to_vtk(
                    filename=aux_path
                    + path
                    + f'/{mat_short}-{geometry[:2].title()}-S{int(self.struct.box[0])}-T{int(thickness)}-P{int(pore_vol_frac*100)}-L{loading:.2f}-ID{counter}-{filename_extra}'
                )

                with open(
                    aux_path
                    + path
                    + f'/{mat_short}-{geometry[:2].title()}-S{int(self.struct.box[0])}-T{int(thickness)}-P{int(pore_vol_frac*100)}-L{loading:.2f}-ID{counter}-{filename_extra}.pkl',
                    'wb',
                ) as f:
                    pickle.dump([self.struct, self.structure_verification, self.data], f)

                # Optional channelization sweep (overrides given wall_th/channel_th)
                if channel:
                    for w_th, c_th in itertools.product(wall_th, channel_th):
                        shape_x = self.struct.box_shape[0]
                        x_indices = list(range(w_th, shape_x, w_th + c_th))
                        for i, x_idx in enumerate(x_indices):
                            end_idx = min(x_idx + c_th, shape_x)
                            self.retain_groups_x(x_idx, end_idx, w_th)
                        self.struct.to_vtk(
                            filename=aux_path
                            + path
                            + f'/{mat_short}-{geometry[:2].title()}-S{int(self.struct.box[0])}-T{int(thickness)}-P{int(pore_vol_frac*100)}-L{loading:.2f}-ID{counter}-{filename_extra}-WTh{w_th}-CTh{c_th}'
                        )

    def pad_structure(self, margins, material: str | None = None):
        """Crop to tight bounding box of label==1 and add padding.

        Parameters
        ----------
        margins : [upper, lower, left/backward, right/front] in µm.
        material : optional string for diagnostics.
        """
        margins_ = [int(m / self.struct.voxel) for m in margins]

        indices = np.argwhere(self.struct.im == 1)
        if len(indices) == 0:
            print(f"No {material or 'material'} found.")
            return self.struct.im

        min_x, min_y, min_z = indices.min(axis=0)
        max_x, max_y, max_z = indices.max(axis=0)

        # Tight crop
        self.struct.im = self.struct.im[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]

        # New padded shape (clamped to ≥1)
        padded_shape = (
            max(1, self.struct.im.shape[0] + margins_[2] + margins_[3]),
            max(1, self.struct.im.shape[1] + margins_[2] + margins_[3]),
            max(1, self.struct.im.shape[2] + margins_[0] + margins_[1]),
        )
        padded_array = np.zeros(padded_shape, dtype=self.struct.im.dtype)

        # Where to paste cropped block
        x_start, x_end = margins_[2], margins_[2] + self.struct.im.shape[0]
        y_start, y_end = margins_[2], margins_[2] + self.struct.im.shape[1]
        z_start, z_end = margins_[0], margins_[0] + self.struct.im.shape[2]

        x_end = min(x_end, padded_shape[0])
        y_end = min(y_end, padded_shape[1])
        z_end = min(z_end, padded_shape[2])

        padded_array[x_start:x_end, y_start:y_end, z_start:z_end] = self.struct.im
        self.struct.im = padded_array.astype(np.uint16)

    def retain_groups_x(self, start_idx: int, end_idx: int, wall_th: int):
        """Zero-out labels in a channel strip along X unless supported by adjacent walls.

        Heuristic: a label must occupy >50% of its (wall + channel) voxels to be retained.
        Also removes any CBD voxels (assumed to be the max label) from the channel.
        """
        wall_start_idx = max(0, start_idx - wall_th)
        wall_region = self.struct.im[wall_start_idx:start_idx, :, :].copy()
        channel_region = self.struct.im[start_idx:end_idx, :, :].copy()
        next_wall_start_idx = end_idx
        next_wall_end_idx = min(self.struct.im.shape[0], end_idx + wall_th)
        next_wall_region = self.struct.im[next_wall_start_idx:next_wall_end_idx, :, :].copy()

        CBD_label = int(np.max(self.struct.im))

        wall_unique_groups = np.unique(wall_region)
        wall_unique_groups = wall_unique_groups[(wall_unique_groups != 0) & (wall_unique_groups != CBD_label)]

        channel_unique_groups = np.unique(channel_region)
        channel_unique_groups = channel_unique_groups[(channel_unique_groups != 0) & (channel_unique_groups != CBD_label)]

        next_wall_unique_groups = np.unique(next_wall_region)
        next_wall_unique_groups = next_wall_unique_groups[(next_wall_unique_groups != 0) & (next_wall_unique_groups != CBD_label)]

        groups_to_delete: List[int] = []

        for group in wall_unique_groups:
            wall_count = int(np.sum(wall_region == group))
            total_count = wall_count + int(np.sum(channel_region == group))
            proportion = wall_count / total_count if total_count > 0 else 0.0
            if proportion <= 0.5:
                groups_to_delete.append(group)

        for group in next_wall_unique_groups:
            next_wall_count = int(np.sum(next_wall_region == group))
            total_count = next_wall_count + int(np.sum(channel_region == group))
            proportion = next_wall_count / total_count if total_count > 0 else 0.0
            if proportion <= 0.5:
                groups_to_delete.append(group)

        for group in groups_to_delete:
            wall_region[wall_region == group] = 0
            channel_region[channel_region == group] = 0
            next_wall_region[next_wall_region == group] = 0

        unique_to_channel = np.setdiff1d(channel_unique_groups, np.union1d(wall_unique_groups, next_wall_unique_groups))
        for group in unique_to_channel:
            channel_region[channel_region == group] = 0

        channel_region[channel_region == CBD_label] = 0

        self.struct.im[wall_start_idx:start_idx, :, :] = wall_region
        self.struct.im[start_idx:end_idx, :, :] = channel_region
        self.struct.im[next_wall_start_idx:next_wall_end_idx, :, :] = next_wall_region

    @staticmethod
    def exceptions(cbd_method, geometry, target_mass_form_am, target_mass_form_binder, target_mass_form_carbon, aux_path, counter):
        """Basic input validation and guardrails."""
        cbd_methods = ['bridge', 'mistry', 'random', 'mixed']
        if cbd_method not in cbd_methods:
            raise IncorrectBinderException(' CBD method not considered.')
        geometries = ['sphere', 'ellipsoid', 'half-sp-half-el']
        if geometry not in geometries:
            raise IncorrectGeometryException(' Geometry not considered.')
        if sum([target_mass_form_am, target_mass_form_binder, target_mass_form_carbon]) > 100:
            raise IncorrectMassFormulation(' Sum of mass ratios is superior to 100 %')

    @staticmethod
    def print_init_info(
        size: int,
        CBD_nanoporosity: float,
        target_mass_form_am: int,
        target_mass_form_binder: int,
        actual_struct: int,
        total_struct: int,
        target_mass_form_carbon: int,
        sph_method: str,
        overlap: float,
        max_tries: int,
        geometry: str,
        cbd_method: str,
        thickness: int,
        target_porosity: float,
    ):
        """Console header for traceability."""
        print('---------------------------------------------------------------------------------------')
        print('')
        print(' Geometry: ', geometry)
        print(' CBD method: ', cbd_method)
        print(' Size: ', size)
        print(' CBD nanoporosity: ', CBD_nanoporosity)
        print(' Mass: ' + str(target_mass_form_am) + '-' + str(target_mass_form_binder) + '-' + str(target_mass_form_carbon))
        print(' Electrode thickness: ', thickness)
        print(f' Target porosity: {target_porosity:.2f}')
        print(' Structure ' + str(actual_struct) + ' of ' + str(total_struct))
        if sph_method == 'gravity_CFD': 
            print(' Overlap: ', overlap)
            print(' Max tries: ', max_tries)

    def generate_structure_add_AM(
        self,
        variable_vox_size: bool,
        thickness: int,
        size: int,
        sph_method: str,
        overlap: float,
        max_tries: int,
        geometry: str,
        x0_orig: float | List[float],
        sigma_orig: float | List[float],
        electrode_generation_mode: str,
        ind_mod: bool,
        voxel: float,
        vol_distrib: bool,
        angle_tolerance_flag: bool = True,
        angle_tolerance: int = 10,
        angles_range_flag: bool = True,
        theta_x: int = 0,
        theta_y: int = 0,
        theta_z: int = 0,
        sph_down: bool = True,
        _1_particle: bool = False,
        diameter: int = 10,
        margin: List[int] = [5, 5, 5, 5],
        mat_short: str = 'Gr',
        max_diam: int = 50,
    ):
        """Create StructureSpheres and populate AM phase according to geometry and CFD."""

        # Create empty structure grid (box in µm; voxel size in µm/voxel)
        self.struct = StructureSpheres(box=[size, size, thickness], voxel=voxel)

        # Set up candidate diameters and (sometimes) volumes per geometry/method
        if geometry == 'sphere' and not _1_particle:
            if sph_method == 'rsa_CFD':
                diams = np.linspace(0.01, max_diam, 1000)
                vols = [4 * np.pi * (diam / 2) ** 2 / 3 for diam in diams]  # unused
            elif sph_method == 'gravity_CFD':
                diams = np.linspace(10, 33, 1000)
                vols = [4 * np.pi * (diam / 2) ** 2 / 3 for diam in diams]  # unused
        elif geometry == 'ellipsoid':
            if vol_distrib:
                factor_y = [random.uniform(0.5, 0.75) for _ in range(1000)]
                factor_z = [random.uniform(0.15, 0.4) for _ in range(1000)]
                diams_sph = np.linspace(0.01, 70, 1000)
                vols = [4 * np.pi * (d_sph / 2) ** 3 / 3 for d_sph in diams_sph]
                diams_x = [2 * (3 * v / (4 * np.pi * factor_y[i] * factor_z[i])) ** (1 / 3) for i, v in enumerate(vols)]
                diams_y = [factor_y[i] * d_x for i, d_x in enumerate(diams_x)]
                diams_z = [factor_z[i] * d_x for i, d_x in enumerate(diams_x)]
                vols_ellipsoid = [4 * np.pi * (d_x / 2) * (diams_y[i] / 2) * (diams_z[i] / 2) / 3 for i, d_x in enumerate(diams_x)]
                vol_similarity = np.mean([v / vols_ellipsoid[i] for i, v in enumerate(vols)])
                print(f" Avg. V_sph/V_ell: {vol_similarity:.2f}")
                diams = diams_x
            else:
                diams_x = np.linspace(0.01, 70, 1000)
                diams_y = np.linspace(0.01, 70, 1000)
                diams_z = [random.uniform(2, 8) for _ in range(1000)]
                diams_y = [d_y * random.uniform(0.55, 0.85) for d_y in diams_y]
                diams = diams_x
        elif geometry == 'half-sp-half-el':
            if sph_method == 'rsa_CFD':
                diams_sp = np.linspace(10, 35, 1000)
                vols = [4 * np.pi * (diam / 2) ** 2 / 3 for diam in diams_sp]  # unused
            elif sph_method == 'gravity_CFD':
                diams_sp = np.linspace(10, 33, 1000)
                vols = [4 * np.pi * (diam / 2) ** 2 / 3 for diam in diams_sp]  # unused
            else:
                raise ValueError('Generation method not considered for spheres.')

            if vol_distrib:
                factor_y = [random.uniform(0.5, 0.75) for _ in range(1000)]
                factor_z = [random.uniform(0.15, 0.4) for _ in range(1000)]
                diams_sph = np.linspace(0.01, 70, 1000)
                vols = [4 * np.pi * (d_sph / 2) ** 3 / 3 for d_sph in diams_sph]
                diams_x = [2 * (3 * v / (4 * np.pi * factor_y[i] * factor_z[i])) ** (1 / 3) for i, v in enumerate(vols)]
                diams_y = [factor_y[i] * d_x for i, d_x in enumerate(diams_x)]
                diams_z = [factor_z[i] * d_x for i, d_x in enumerate(diams_x)]
                vols_ellipsoid = [4 * np.pi * (d_x / 2) * (diams_y[i] / 2) * (diams_z[i] / 2) / 3 for i, d_x in enumerate(diams_x)]
                print(f" Avg. V_sph/V_ell: {np.mean([v / vols_ellipsoid[i] for i, v in enumerate(vols)]):.2f}")
            else:
                diams_x = np.linspace(0.01, 70, 1000)
                diams_y = np.linspace(0.01, 70, 1000)
                diams_z = [random.uniform(2, 8) for _ in range(1000)]
                diams_y = [d_y * random.uniform(0.55, 0.85) for d_y in diams_y]

        # Insert particles according to geometry/method
        if geometry == 'sphere':
            if not _1_particle:
                cfd = cfd_lognormal(diams, x0_orig, sigma_orig)
                # Normalize CFD to [0,1]
                cfd = (cfd - np.min(cfd)) / (np.max(cfd) - np.min(cfd))
                cfd = np.row_stack(cfd).T
            if sph_method == 'rsa_CFD':
                if _1_particle:
                    self.struct.add_one_sphere(diameter=diameter)
                else:
                    self.data['Input parameters']['Overlap'] = overlap
                    if ind_mod:
                        self.struct.add_spheres_rsa_cfd_individual(
                            target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                            diams=diams,
                            cfd=cfd,
                            ratios=[1.0],
                            overlap=overlap,
                            electrode_generation_mode=electrode_generation_mode,
                        )
                    else:
                        self.struct.add_spheres_rsa_cfd(
                            target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                            diams=diams,
                            cfd=cfd,
                            ratios=[1.0],
                            overlap=overlap,
                            electrode_generation_mode=electrode_generation_mode,
                        )
            elif sph_method == 'gravity_CFD':
                self.data['Input parameters']['Overlap'] = overlap
                self.axis = 2  # 0:x, 1:y, 2:z
                self.struct.add_spheres_gravity_cfd_counted(
                    target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                    diams=diams,
                    cfd=cfd,
                    overlap=overlap,
                )
            else:
                print(' Method not considered')

        elif geometry == 'ellipsoid':
            cfd = cfd_lognormal(diams_x, x0_orig, sigma_orig)
            cfd = (cfd - np.min(cfd)) / (np.max(cfd) - np.min(cfd))
            cfd = np.row_stack(cfd).T
            if _1_particle:
                self.struct.add_one_ellipsoid(diams=diams, cfd=cfd, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
            elif sph_method == 'rsa_CFD':
                self.data['Input parameters']['Overlap'] = overlap
                if ind_mod:
                    self.struct.add_ellipsoids_rsa_cfd_counted_individual(
                        target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                        diams=[diams_x, diams_y, diams_z],
                        cfd=cfd,
                        ratios=None,
                        overlap=overlap,
                        electrode_generation_mode=electrode_generation_mode,
                        angle_tolerance_flag=angle_tolerance_flag,
                        angle_tolerance=angle_tolerance,
                        angles_range_flag=angles_range_flag,
                        theta_x=theta_x,
                        theta_y=theta_y,
                        theta_z=theta_z,
                    )
                else:
                    self.struct.add_ellipsoids_rsa_cfd(
                        target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                        diams=[diams_x, diams_y, diams_z],
                        cfd=cfd,
                        ratios=None,
                        overlap=overlap,
                        electrode_generation_mode=electrode_generation_mode,
                        angle_tolerance_flag=angle_tolerance_flag,
                        angle_tolerance=angle_tolerance,
                        angles_range_flag=angles_range_flag,
                        theta_x=theta_x,
                        theta_y=theta_y,
                        theta_z=theta_z,
                    )
            elif sph_method == 'gravity_CFD':
                if ind_mod:
                    self.data['Input parameters']['Overlap'] = overlap
                    self.axis = 2
                    self.struct.add_ellipsoids_gravity_cfd_counted_individual(
                        target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                        diams=[diams_x, diams_y, diams_z],
                        cfd=cfd,
                        ratios=None,
                        overlap=overlap,
                        electrode_generation_mode=electrode_generation_mode,
                        angle_tolerance_flag=angle_tolerance_flag,
                        angle_tolerance=angle_tolerance,
                        angles_range_flag=angles_range_flag,
                        theta_x=theta_x,
                        theta_y=theta_y,
                        theta_z=theta_z,
                        max_tries=max_tries,
                        axis=self.axis,
                    )

        elif geometry == 'half-sp-half-el':
            self.data['Input parameters']['Overlap'] = overlap
            cfd_sp = cfd_lognormal(diams_sp, x0_orig[0], sigma_orig[0])
            cfd_sp = (cfd_sp - np.min(cfd_sp)) / (np.max(cfd_sp) - np.min(cfd_sp))
            cfd_sp = np.row_stack(cfd_sp).T
            cfd_el = cfd_lognormal(diams_x, x0_orig[1], sigma_orig[1])
            cfd_el = (cfd_el - np.min(cfd_el)) / (np.max(cfd_el) - np.min(cfd_el))
            cfd_el = np.row_stack(cfd_el).T
            self.struct.add_mixed_geom_rsa_cfd_individual(
                target_porosity=(1.0 - self.data['Calculated parameters'][f'{mat_short} volume fraction']),
                diams=[diams_sp, diams_x, diams_y, diams_z],
                cfd=[cfd_sp, cfd_el],
                ratios=None,
                overlap=overlap,
                electrode_generation_mode=electrode_generation_mode,
                angle_tolerance_flag=angle_tolerance_flag,
                angle_tolerance=angle_tolerance,
                angles_range_flag=angles_range_flag,
                theta_x=theta_x,
                theta_y=theta_y,
                theta_z=theta_z,
                z_ratio=0.25,
                sph_down=True,
            )
        else:
            raise ValueError(' Geometry not considered')

    def cutting(
        self,
        path: str,
        sph_method: str,
        cut: bool,
        thickness: int,
        size: int,
        diam: list,
        target_mass_form_am: int,
        target_mass_form_binder: int,
        target_mass_form_carbon: int,
        dens_am: float,
        dens_binder: float,
        dens_carbon: float,
        CBD_nanoporosity: float,
        target_porosity: float,
        aux_path: str,
        mat_short: str,
        axis: int = 0,
    ):
        """Optionally cut a gravity‑packed electrode along a chosen axis and recompute targets."""
        if path == 'NoCBD/GoodStructure/':
            if sph_method == 'gravity_CFD' and cut is True:
                cut_axis = ['x', 'y', 'z'][axis] if axis in {0, 1, 2} else None
                elec_to_cut = self.struct.cut_electrode(self.struct.im, diam[-1], cut_axis)
                for i in range(3):
                    self.struct.box[i] = float(elec_to_cut.shape[i])
                self.struct.box_shape = (self.struct.box / self.struct.voxel).astype(int)
                self.struct.im = elec_to_cut
                Lx, Ly, thickness = self.struct.box[0], self.struct.box[1], self.struct.box[2]

                self.formula = FormulationDry(
                    mass_am=target_mass_form_am,
                    mass_binder=target_mass_form_binder,
                    mass_carbon=target_mass_form_carbon,
                    dens_am=dens_am,
                    dens_binder=dens_binder,
                    dens_carbon=dens_carbon,
                    CBD_nanoporosity=CBD_nanoporosity,
                    vol=[Lx, Ly, thickness],
                    target_porosity=target_porosity,
                )

                new_vol_frac_am = self.formula.data['am']['vol_frac']
                new_vol_frac_CBD = self.formula.data['CBD']['vol_frac']

                self.data['Calculated parameters'][f'{mat_short} volume fraction'] = new_vol_frac_am
                self.data['Calculated parameters']['CBD volume fraction'] = new_vol_frac_CBD
                new_target_porosity = 1 - (new_vol_frac_CBD + new_vol_frac_am)
                self.data['Input parameters']['Target porosity'] = new_target_porosity
            else:
                # No cut: mirror the current targets into data dict
                new_vol_frac_am = self.formula.data['am']['vol_frac']
                new_vol_frac_CBD = self.formula.data['CBD']['vol_frac']
                self.data['Calculated parameters'][f'{mat_short} volume fraction'] = new_vol_frac_am
                self.data['Calculated parameters']['CBD volume fraction'] = new_vol_frac_CBD
                self.data['Input parameters']['Target porosity'] = 1 - (new_vol_frac_CBD + new_vol_frac_am)

    def CBD_addition_erosion(self, cbd_method: str, CBD_nanoporosity: float, vol_frac_CBD: float, ind_mod: bool):
        """Add CBD via chosen algorithm and erode to the target accounting for nanoporosity."""
        CBD_init = time.time()

        binder_label = int(np.max(self.struct.im) + 1) if ind_mod else 2

        vol_mult = 1.0 / (1.0 - CBD_nanoporosity)  # inflate to account for nanoporosity
        if cbd_method == 'bridge':
            self.struct.add_binder_bridge(vol_mult * vol_frac_CBD, increment=0.0001, binder_label=binder_label)
        elif cbd_method == 'mistry':
            self.struct.add_binder_mistry(
                vol_mult * vol_frac_CBD,
                weight=0.01,
                pre_fraction=0.001,
                multiplier=100,
                binder_label=binder_label,
            )
        elif cbd_method == 'random':
            self.struct.add_binder_random(vol_mult * vol_frac_CBD, weight=0.01, pre_fraction=0.001, multiplier=100)
        elif cbd_method == 'blob':
            self.struct.add_binder_blob(vol_mult * vol_frac_CBD, increment=0.05, binder_label=binder_label)
        else:
            print(' CBD method not considered')

        if ind_mod:
            self.struct.erode(phase=binder_label, inrelationto=0, target_vol_frac=vol_mult * vol_frac_CBD)
        else:
            self.struct.erode(phase=2, inrelationto=0, target_vol_frac=vol_mult * vol_frac_CBD)

        print("\n CBD time: {:.2f}".format(time.time() - CBD_init))

    def plot_particles_sizes_distrib_check(
        self,
        path: str,
        thick: int,
        x0_orig: float,
        sigma_orig: float,
        gr: List[float],
        counter: int,
        geometry: str = '',
    ):
        """Plot generated PSD vs reference lognormal and save PNG to `path`.

        Parameters
        ----------
        path : str
            Output directory (must exist).
        thick : int
            Electrode thickness in µm (for title/filename).
        x0_orig, sigma_orig : float
            Reference lognormal params.
        gr : list[float]
            Measured diameters.
        counter : int
            ID used in filename.
        geometry : str
            Optional tag in filename.
        """
        plt.hist([gr], bins=20, alpha=0.4, stacked=True, density=True, label=['Graphite'])
        filename = f"thickness{thick:.2f}um_porosity{self.struct.statistics['porosity']:.2f}_ID{counter}_{geometry}.png"

        xfit = np.linspace(int(min(gr)), round(max(gr)), 1000)
        plt.plot(xfit, lognormal(xfit, x0_orig, sigma_orig), lw=3, label='SEM Distribution')
        plt.xlabel(r'Particle Diameter ($\mu$m)')
        plt.ylabel('Probability')
        plt.title(f"Thickness: {thick:.2f} $\mu$m, porosity: {self.struct.statistics['porosity']:.2f}")
        plt.legend()
        print(path + filename)
        plt.savefig(path + filename)
        plt.show()
        plt.close()

    @staticmethod
    def check_diam_distribution_v2(gr: List[float], x0_orig: float, sigma_orig: float, geometry: str):
        """Fit a lognormal to `gr` and compare parameters with reference (x0,sigma).

        Returns a folder hint: 'NoCBD/GoodStructure/' or 'NoCBD/BadStructure/'.
        Tolerances: x0 within ±5%; sigma within −20%/ +20%.
        """
        y, edges = np.histogram(gr, bins=15)
        x = np.convolve(edges, [0.5, 0.5], mode='valid')
        y = y / np.trapz(y, x)  

        expected = (x0_orig, sigma_orig)
        params, cov = optimize.curve_fit(lognormal, x, y, expected)

        if (params[0] > x0_orig * 1.05 or params[0] < x0_orig * 0.95) or (
            params[1] > sigma_orig * 1.2 or params[1] < sigma_orig * 0.8
        ):
            path = 'NoCBD/BadStructure/'
        else:
            path = 'NoCBD/GoodStructure/'

        if not os.path.exists(path):
            os.makedirs(path)

        return path, params[0] / x0_orig, params[1] / sigma_orig

    @staticmethod
    def input_calculated_data_2dict(input_data_list: list, calculated_data_list: list, material: str):
        """Pack input and calculated parameters into a single dictionary."""
        data = {'Input parameters': {}, 'Calculated parameters': {}}
        in_keys = [
            'CBD nanoporosity',
            'Target porosity',
            f'{material} capacity (mAh/g)',
            'Thickness (um)',
            f'Mass {material} (g)',
            'Mass binder (g)',
            'Mass carbon (g)',
        ]
        for i, key in enumerate(in_keys):
            data['Input parameters'][key] = input_data_list[i]

        calc_keys = [f'{material} volume fraction', 'CBD volume fraction']
        for i, key in enumerate(calc_keys):
            data['Calculated parameters'][key] = calculated_data_list[i]
        return data

    # Obtain statistics of the generated electrode
    def obtain_stats(
        self,
        sph_method: str,
        CBD_nanoporosity: float,
        vol: float,
        cap_th: float,
        rho_am: float,
        vol_frac_list: list,
        new_vol_frac_list: list,
        ind_mod: bool,
        mat_short: str,
    ):
        """Compute generated AM/CBD/porosity fractions and mass/capacity.

        Returns
        -------
        [am_vf, cbd_vf, pore_vf, tot_vf], [mass_am_in_box, total_capacity]
        """
        cbd_label = int(np.max(self.struct.im)) if ind_mod else 2
        gen_str_am_vol_frac = self.struct.vol_frac((self.struct.im != 0) & (self.struct.im != cbd_label))
        gen_str_cbd_vol_frac = (1.0 - CBD_nanoporosity) * self.struct.vol_frac(self.struct.im == cbd_label)
        pore_vol_frac = self.struct.vol_frac(self.struct.im == 0) + CBD_nanoporosity * self.struct.vol_frac(
            self.struct.im == cbd_label
        )
        tot_vol_frac = gen_str_am_vol_frac + gen_str_cbd_vol_frac

        [vol_frac_am, vol_frac_CBD, target_porosity] = vol_frac_list
        [new_vol_frac_am, new_vol_frac_CBD, new_target_porosity] = new_vol_frac_list

        if sph_method == 'rsa_CFD':
            print(' Volume fractions of the generated microstructures (eliminating nanoporosity) ')
            print(f' {mat_short} domain volume fraction: {gen_str_am_vol_frac}')
            print(f' CBD domain volume fraction: {gen_str_cbd_vol_frac}')
            print(f' Porosity: {pore_vol_frac}')
            print(f' Sum: {gen_str_am_vol_frac + gen_str_cbd_vol_frac + pore_vol_frac}')
            print('------------------------------------------------------------------------')
            print(' Volume fractions calculated from inserted formulation ')
            print(f' {mat_short} domain volume fraction: {vol_frac_am}')
            print(f' CBD domain volume fraction: {vol_frac_CBD}')
            print(f' Porosity: {target_porosity}')
            print(f' Sum: {vol_frac_am + vol_frac_CBD + target_porosity}')
            print('------------------------------------------------------------------------')
            print(f' {mat_short} vol. ratio: {gen_str_am_vol_frac/vol_frac_am}')
            print(f' CBD vol. ratio: {gen_str_cbd_vol_frac/vol_frac_CBD}')
        elif sph_method == 'gravity_CFD':
            print(' Volume fractions of the generated microstructures (eliminating nanoporosity) ')
            print(f' {mat_short} domain volume fraction: {gen_str_am_vol_frac}')
            print(f' CBD domain volume fraction: {gen_str_cbd_vol_frac}')
            print(f' Porosity: {pore_vol_frac}')
            print(f' Sum: {gen_str_am_vol_frac + gen_str_cbd_vol_frac + pore_vol_frac}')
            print('------------------------------------------------------------------------')
            print(' Volume fractions calculated from inserted formulation ')
            print(f' {mat_short} domain volume fraction: {new_vol_frac_am}')
            print(f' CBD domain volume fraction: {new_vol_frac_CBD}')
            print(f' Porosity: {new_target_porosity}')
            print(f' Sum: {new_vol_frac_am + new_vol_frac_CBD + new_target_porosity}')

        mass_am_in_box = vol * gen_str_am_vol_frac * rho_am
        totcap = mass_am_in_box * cap_th

        return [gen_str_am_vol_frac, gen_str_cbd_vol_frac, pore_vol_frac, tot_vol_frac], [mass_am_in_box, totcap]

    def output_data2dict(
        self,  
        output_vol_frac_list: list,
        vol_frac_list: list,
        new_vol_frac_list: list,
        vol: float,
        rho_list: list,
        m_list: list,
        area_list: list,
        duration: float,
        cm2toum2: float,  
        totcap: float,
        sph_method: str,
        aux_path: str,
        data: dict,
        mat_short: str,
    ):
        """Assemble verification dicts, compute areal loading, and decide output path.

        Returns
        -------
        loading, data, structure_verification, path
        """
        structure_verification = {
            'Vol. fract. generated ustruct': {},
            'Vol. fract. calculated from inserted formulation': {},
        }
        structure_verification['Vol. fract. generated ustruct'][f'{mat_short} domain volume fraction'] = output_vol_frac_list[0]
        structure_verification['Vol. fract. generated ustruct']['CBD domain volume fraction'] = output_vol_frac_list[1]
        structure_verification['Vol. fract. generated ustruct']['Porosity'] = output_vol_frac_list[2]

        if sph_method == 'rsa_CFD':
            structure_verification['Vol. fract. calculated from inserted formulation'][f'{mat_short} domain volume fraction'] = vol_frac_list[0]
            structure_verification['Vol. fract. calculated from inserted formulation']['CBD domain volume fraction'] = vol_frac_list[1]
            structure_verification['Vol. fract. calculated from inserted formulation']['Porosity'] = vol_frac_list[2]
            path = self.check_vol_factors_v2(output_vol_frac_list[0], output_vol_frac_list[1], vol_frac_list[0], vol_frac_list[1])
        elif sph_method == 'gravity_CFD':
            structure_verification['Vol. fract. calculated from inserted formulation'][f'{mat_short} domain volume fraction'] = new_vol_frac_list[0]
            structure_verification['Vol. fract. calculated from inserted formulation']['CBD domain volume fraction'] = new_vol_frac_list[1]
            structure_verification['Vol. fract. calculated from inserted formulation']['Porosity'] = new_vol_frac_list[2]
            path, [gen_gr_percentage, gen_CBD_percentage] = self.check_vol_factors_gravity_cfd(
                [output_vol_frac_list[0], output_vol_frac_list[1]], vol, rho_list, m_list
            )
            print(f" Gen. struct. {mat_short}. mass formulation (%)", gen_gr_percentage)
            print(' Gen. struct. CBD mass formulation (%)', gen_CBD_percentage)

        if not os.path.exists(aux_path + path):
            os.makedirs(aux_path + path)

        loading = totcap * cm2toum2 / (area_list[0] * area_list[1])

        data['Output parameters'] = {
            'Total capacity (mAh)': totcap,
            'Loading (mAh/um2)': loading,
            'Duration (min)': duration,
        }

        print(f' Capacity: {totcap}')
        print(f' Loading: {loading:.4f}')
        print('')
        print(' ' + path)

        return loading, data, structure_verification, path

    @staticmethod
    def check_vol_factors_v2(gen_str_am_vol_frac, gen_str_cbd_vol_frac, vol_frac_am, vol_frac_cbd):
        """Check AM/CBD volume fractions within ±5% tolerance vs targets.

        Returns folder hint 'CBD/GoodStructure' or 'CBD/BadStructure'.
        """
        if (gen_str_am_vol_frac > 1.05 * vol_frac_am or gen_str_am_vol_frac < 0.95 * vol_frac_am):
            path = 'CBD/BadStructure'
        else:
            path = 'CBD/GoodStructure'
        return path

    @staticmethod
    def check_vol_factors_gravity_cfd(gen_vol_frac: list, vol: float, rho_list: list, m_list: list):
        """Verify mass fractions for gravity‑CFD case using generated volume fractions and densities.

        """
        mGr_out = gen_vol_frac[0] * vol * rho_list[0]
        mCBD_out = gen_vol_frac[1] * vol * rho_list[1]

        gen_gr_percentage = mGr_out / (mGr_out + mCBD_out) * 100
        gen_CBD_percentage = mCBD_out / (mGr_out + mCBD_out) * 100

        if (gen_gr_percentage > 1.05 * 96 or gen_gr_percentage < 0.95 * 90) or (
            gen_CBD_percentage > 1.05 * 10 or gen_CBD_percentage < 0.95 * 4
        ):
            path = 'CBD/BadStructure'
        else:
            path = 'CBD/GoodStructure'
        return path, [gen_gr_percentage, gen_CBD_percentage]


# ----------------------------- Exceptions ------------------------------------
class IncorrectBinderException(Exception):
    pass


class IncorrectGeometryException(Exception):
    pass


class IncorrectMassFormulation(Exception):
    pass


class IncorrectParamTestSet(Exception):
    pass
