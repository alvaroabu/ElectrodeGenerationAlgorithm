import numpy as np
from numba import njit
import random

__all__ = [
    '_make_ellipse',
    '_make_ellipsoid',
    '_insert_ellipse_at_points',
    '_insert_ellipse_at_points_periodic',
    '_insert_ellipses_at_points'
]

@njit(parallel=False)
def _insert_ellipse_at_points(im, coords, rA, rB, rC, v,
                            smooth=True, overwrite=False,
                            rotation_angles=False, 
                            theta_x=0, theta_y=0, theta_z=0):  # pragma: no cover
    r"""
    Insert ellipsoids (or ellipses) into the given ND-image at given locations

    This function uses numba to accelerate the process, and does not
    overwrite any existing values (i.e. only writes to locations containing
    zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the ellipsoids/ellipses should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each ellipsoids/ellipses in an array of shape
        ``ndim by npts``
    rA, rB, rC : int
        The radius of all the ellipsoids/ellipses to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the ellipsoids/ellipses will not have the litte
        nibs on the surfaces.
    rotation_angles : boolean
        If ``False`` (default) then the rotation angles will be random.
        If ``True`` then the rotation angles will be given.
    theta_x, theta_y, theta_z : int
        The values of the rotation angles for each axes in radians. By default
        they are set to 0. If rotation_angles is not activated, the angles are
        randomly calculated.
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        s, angles = _make_ellipse(rA, rB, smooth=True, rotation_angles=rotation_angles, 
                          theta_x=theta_x)        
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                        if (y >= 0) and (y < ylim):
                            if s[a, b] == 1:
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        s, angles = _make_ellipsoid(rA, rB, rC, smooth=True, rotation_angles=rotation_angles,
                                    theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-s.shape[2]//2, pt[2]+s.shape[2]//2-1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im, angles

@njit
def _insert_ellipse_at_points_periodic(im, coords, rA, rB, rC, v,
                                       smooth=True, overwrite=False, 
                                       rotation_angles=False, 
                                       theta_x=0, theta_y=0, theta_z=0):  # pragma: no cover
    r"""
    Insert ellipsoids (or ellipse) into the given ND-image at given locations

    This function uses numba to accelerate the process, and does not
    overwrite any existing values (i.e. only writes to locations containing
    zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the ellipsoids/ellipses should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each ellipsoids/ellipses in an array of shape
        ``ndim by npts``
    rA, rB, rC : int
        The radius of all the ellipsoids/ellipses to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the ellipsoids/ellipses will not have the litte
        nibs on the surfaces.
    rotation_angles : boolean
        If ``False`` (default) then the rotation angles will be random.
        If ``True`` then the rotation angles will be given.
    theta_x, theta_y, theta_z : int
        The values of the rotation angles for each axes in radians. By default
        they are set to 0. If rotation_angles is not activated, the angles are
        randomly calculated.
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        s = _make_ellipse(rA, rB, smooth=True, rotation_angles=rotation_angles, 
                          theta_x=theta_x) 
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if x < 0:
                    x = x + xlim
                elif x >= xlim:
                    x = x - xlim
                for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                    if y < 0:
                        y = y + ylim
                    elif y >= ylim:
                        y = y - ylim
                    if s[a, b] == 1:
                        if overwrite or (im[x, y] == 0):
                            im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        s, angles = _make_ellipsoid(rA, rB, rC, smooth=True, rotation_angles=rotation_angles,
                                    theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)     
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if x < 0:
                    x = x + xlim
                elif x >= xlim:
                    x = x - xlim
                for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                    if y < 0:
                        y = y + ylim
                    elif y >= ylim:
                        y = y - ylim
                    for c, z in enumerate(range(pt[2]-s.shape[2]//2, pt[2]+s.shape[2]//2-1)):
                        if z < 0:
                            z = z + zlim
                        elif z >= zlim:
                            z = z - zlim
                        if (s[a, b, c] == 1):
                            if overwrite or (im[x, y, z] == 0):
                                im[x, y, z] = v
    return im, angles

@njit(parallel=False)
def _insert_ellipses_at_points(im, coords, radii, v, smooth=True,
                               overwrite=False, rotation_angles=False, 
                               theta_x=0, theta_y=0, theta_z=0):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean, optional
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.
    rotation_angles : boolean
        If ``False`` (default) then the rotation angles will be random.
        If ``True`` then the rotation angles will be given.
    theta_x, theta_y, theta_z : int
        The values of the rotation angles for each axes in radians. By default
        they are set to 0. If rotation_angles is not activated, the angles are
        randomly calculated.
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            pt = coords[:, i]
            rA, rB = radii[i]
            s = _make_ellipse(rA, rB, smooth=True, rotation_angles=rotation_angles, 
                  theta_x=theta_x)         
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if x < 0:
                    x = x + xlim
                elif x >= xlim:
                    x = x - xlim
                for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                    if y < 0:
                        y = y + ylim
                    elif y >= ylim:
                        y = y - ylim
                    if s[a, b] == 1:
                        if overwrite or (im[x, y] == 0):
                            im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            pt = coords[:, i]
            rA, rB, rC = radii[i]
            s, angles = _make_ellipsoid(rA, rB, rC, smooth=True, rotation_angles=rotation_angles,
                                        theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
            for a, x in enumerate(range(pt[0]-s.shape[0]//2, pt[0]+s.shape[0]//2-1)):
                if x < 0:
                    x = x + xlim
                elif x >= xlim:
                    x = x - xlim
                for b, y in enumerate(range(pt[1]-s.shape[1]//2, pt[1]+s.shape[1]//2-1)):
                    if y < 0:
                        y = y + ylim
                    elif y >= ylim:
                        y = y - ylim
                    for c, z in enumerate(range(pt[2]-s.shape[2]//2, pt[2]+s.shape[2]//2-1)):
                        if z < 0:
                            z = z + zlim
                        elif z >= zlim:
                            z = z - zlim
                        if (s[a, b, c] == 1):
                            if overwrite or (im[x, y, z] == 0):
                                im[x, y, z] = v
    return im, angles

@njit(parallel=False)
def _make_ellipse(rA, rB, smooth=True, rotation_angles=False, theta_x=0):  # pragma: no cover
    r"""
    Generate a ellipse structuring element of the given radius

    Parameters
    ----------
    rA, rB : int
        The radius of the desired ellipse
    smooth : boolean
        If ``True`` (default) then the ellipse will not have the litte
        nibs on the surfaces.
    rotation_angles : boolean
        If ``False`` (default) then the rotation angles will be random.
        If ``True`` then the rotation angles will be given.
    theta_x: int
        The value of the rotation angle for x axes in radians. By default
        they are set to 0. If rotation_angles is not activated, the angles are
        randomly calculated.
    Returns
    -------
    ellipse : ndarray
        A numpy array of 1 and 0 suitable for use as a structuring element
    """
    
    # Set threshold
    if smooth:
        thresh = 1 - 0.001
    else:
        thresh = 1
     
    max_dimension = max(rA, rB)    
    grid_size = int(2 * max_dimension + 1)  # Ensure the grid covers the entire ellipse after rotation
    
    # Create a grid of points
    x_range = np.linspace(-max_dimension, max_dimension+1, grid_size)
    y_range = np.linspace(-max_dimension, max_dimension+1, grid_size)

    # Due to incompatibility with numba, I use nested for loops
    # x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
   
    x_grid = np.zeros((2*max_dimension+1, 2*max_dimension+1))
    y_grid = np.zeros((2*max_dimension+1, 2*max_dimension+1))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            x_grid[i,j] = x
            y_grid[i,j] = y
         
    ellipse = (x_grid**2 / rA**2) + (y_grid**2 / rB**2) <= thresh        
    
    if rotation_angles == False:
        # Define rotation angle (in radians) 
        theta_x = random.uniform(0, 2 * np.pi)

    # Rotation matrices
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)

    # Apply rotation to coordinates
    rotated_x = x_grid*cos_x - y_grid*sin_x
    rotated_y = x_grid*sin_x + y_grid*cos_x
    
    # Get rotated ellipsoid
    rotated_ellipse = (rotated_x**2 / rA**2) + (rotated_y**2 / rB**2) <= thresh

    values = np.where(rotated_ellipse)
    x_low = min(values[0]) - 3 
    x_high = max(values[0]) + 3
    y_low = min(values[1]) - 3
    y_high = max(values[1]) + 3

    # Use the highest and lowest indexes to slice the array
    sliced_array = rotated_ellipse[max(x_low,0):min(x_high, rotated_ellipse.shape[0]),
                                   max(y_low,0):min(y_high, rotated_ellipse.shape[1])]
    s = 1*sliced_array
    
    return s, [theta_x]


@njit(parallel=False)
def _make_ellipsoid(rA, rB, rC, smooth=True, rotation_angles=False, theta_x=0,
                    theta_y=0, theta_z=0):  # pragma: no cover
    r"""
    Generate a spherical structuring element of the given radius

    Parameters
    ----------
    rA, rB, rC : int
        The radius of the desired ellipsoid
    smooth : boolean
        If ``True`` (default) then the ellipsoid will not have the litte
        nibs on the surfaces.
    rotation_angles : boolean
        If ``False`` (default) then the rotation angles will be random.
        If ``True`` then the rotation angles will be given.
    theta_x, theta_y, theta_z : int
        The values of the rotation angles for each axes in radians. By default
        they are set to 0. If rotation_angles is not activated, the angles are
        randomly calculated.
    Returns
    -------
    ball : ndarray
        A numpy array of 1 and 0 suitable for use as a structuring element
    """
    # Set threshold
    if smooth:
        thresh = 1 - 0.001
    else:
        thresh = 1
        
    max_dimension = max(rA, rB, rC)
    grid_size = int(2 * max_dimension + 1)  # Ensure the grid covers the entire ellipse after rotation
    
    # Create a grid of points
    x_range = np.linspace(-max_dimension, max_dimension+1, grid_size)
    y_range = np.linspace(-max_dimension, max_dimension+1, grid_size)
    z_range = np.linspace(-max_dimension, max_dimension+1, grid_size)

    # Due to incompatibility with numba, I use nested for loops
    # x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
   
    x_grid = np.zeros((2*max_dimension+1, 2*max_dimension+1, 2*max_dimension+1))
    y_grid = np.zeros((2*max_dimension+1, 2*max_dimension+1, 2*max_dimension+1))
    z_grid = np.zeros((2*max_dimension+1, 2*max_dimension+1, 2*max_dimension+1))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, z in enumerate(z_range):
                x_grid[i,j,k] = x
                y_grid[i,j,k] = y
                z_grid[i,j,k] = z
         
    # ellipsoid = (x_grid**2 / rA**2) + (y_grid**2 / rB**2) + (z_grid**2 / rC**2) <= thresh        
    
    if rotation_angles == False:
    # Define rotation angle (in radians) 
        theta_x = random.uniform(0, 2 * np.pi)
        # theta_y = 0
        theta_y = random.uniform(0, 2 * np.pi)
        # theta_z = 0
        theta_z = random.uniform(0, 2 * np.pi)

    # Apply rotation to coordinates
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)

    x1 = x_grid
    y1 = y_grid*cos_x - z_grid*sin_x
    z1 = y_grid*sin_x + z_grid*cos_x

    x2 = x1*cos_y + z1*sin_y 
    y2 = y1
    z2 = -x1*sin_y + z1*cos_y

    rotated_x = x2*cos_z - y2*sin_z
    rotated_y = x2*sin_z - y2*cos_z
    rotated_z = z2   
    
    # Get rotated ellipsoid
    rotated_ellipsoid = (rotated_x**2 / rA**2) + (rotated_y**2 / rB**2) + (rotated_z**2 / rC**2) <= thresh

    values = np.where(rotated_ellipsoid)
    x_low = min(values[0]) - 3 
    x_high = max(values[0]) + 3
    y_low = min(values[1]) - 3
    y_high = max(values[1]) + 3
    z_low = min(values[2]) - 3
    z_high = max(values[2]) + 3

    # Use the highest and lowest indexes to slice the array
    sliced_array = rotated_ellipsoid[max(x_low,0):min(x_high, rotated_ellipsoid.shape[0]),
                                     max(y_low,0):min(y_high, rotated_ellipsoid.shape[1]),
                                     max(z_low,0):min(z_high, rotated_ellipsoid.shape[2])]
    s = 1*sliced_array
  
    # import porespy as ps
    # ps.io.to_vtk(s, filename="C:/Users/aabucide/OneDrive - CIC energiGUNE/Escritorio/Work/EllipsoidalStructuresGeneration/s" + str(np.degrees(theta_x)) + '-' + str(np.degrees(theta_y)) + '-' + str(np.degrees(theta_z)))
    
    # import matplotlib.pyplot as plt
    # for i in range(s.shape[2]):
    #     plt.imshow(s[:,:,i])
    #     plt.show()
    #     plt.close()
        
    # for i in range(ellipsoid.shape[2]):
    #     plt.imshow(ellipsoid[:,:,i])
    #     plt.show()
    #     plt.close()
        
    return s, [theta_x, theta_y, theta_z]