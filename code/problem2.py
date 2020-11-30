import os

import numpy as np
import matplotlib.pyplot as plt

def index_closest_grid_points(x, grid):
    """Find index of closed grind point for point 'x' given grid 'grid'.

    Hence, only valid for integer grid.

    Args:
        x (float): x coordinate
        grid (list): list intergers containing grid

    Returns:
        tuple: grid points closest to x.
    """
    return round(x)-1, round(x)%len(grid)

def distance(x1, x2, l):
    """Distance between x1 and x2 with periodic boundry conditions for a grid of
    length l

    Args:
        x1 (float): x1 coordinate
        x2 (float): x2 coordinate
        l (float): length of grid

    Returns:
        float: distance between x1 and x2
    """
    dx = abs(x1-x2)
    if dx > 0.5*l:
        return l-dx
    else:
        return dx

def make_3D_density_contrast_grid(positions, shape):
    """Calculate 3D density contrast grid for set of points. Shape denotes the 
    size of the grid. Interpolation is used to distribute the mass between the 
    grid points. The interpolation conserves the total mass.

    Args:
        positions (list(N,3)): List containing coordinates of N points.
        shape (int): Shape of grid.

    Returns:
        numpy.array: array containg density contrast in (shape,shape,shape).
    """
    density_grid = np.zeros(shape)
    # define grid center coordinates
    grid = [[i+0.5 for i in range(j)] for j in shape]
    # Calculate mean density
    mean_density = len(positions)/(shape[0]*shape[1]*shape[2])
    for pos in positions:
        # Obtain closed grid points in x-, y- and z-direction
        grid_point_indeces = [
            index_closest_grid_points(coord, g) for g, coord in zip(grid,pos)
        ]
        pos = [
            grid[i]-len(grid[i]) if pos[i]> len(grid[i]) + 1 else pos[i] 
            for i in range(len(pos))
        ]

        # Loop over gridpoints and distribute mass
        for i in grid_point_indeces[0]:
            for j in grid_point_indeces[1]:
                for k in grid_point_indeces[2]:                    
                    massfrac = \
                        (1-distance(pos[0], grid[0][i], shape[0]))\
                        * (1-distance(pos[1], grid[1][j], shape[1]))\
                        * (1-distance(pos[2], grid[2][k], shape[2]))
                    density_grid[i, j, k] += massfrac
    # Convert density grid to density contrast
    density_contrast = (density_grid - mean_density)/mean_density
    return density_contrast

def dft(x, Nj, inverse=False):
    """Direct Fourier Transform used by fft function. Implementation according 
    to Cooley-Tukey algorithm. Implementation is non-inplace.

    Args:
        x (list): values to Fourier transform
        Nj (int): length of x
        inverse (bool, optional): If True, perform inverse Fourier Transform. 
        Defaults to False.

    Returns:
        list: FT[x] or FT^-1[x]
    """
    if Nj == 1:
        return x
    else:
        # Recursive call to dft for even and odd indeces of x.
        x = np.append(
            dft(
                x=x[0::2],
                Nj=Nj/2,
                inverse=inverse
            ),
            dft(
                x=x[1::2], 
                Nj=Nj/2,
                inverse=inverse
            )
        )
    
        W_Nj = np.exp(2j*np.pi/Nj)
        # if ineverse add minus sing in exponent
        if inverse:
            W_Nj = W_Nj**-1

        # combine even and odd part of x using W_Nj
        for k in range(int(Nj/2)):
            t = x[k]
            x[k] = t + W_Nj**k * x[k+int(Nj/2)]
            x[k+int(Nj/2)] = t - W_Nj**k * x[k+int(Nj/2)]
        return x

def fft(x, inverse=False):
    """Fast Fourier Transform, Cooley-Tukey algorithm. Non-inplace 
    implementation.

    Args:
        x (list): Values to Fourier Transform
        inverse (bool, optional): If True, perform inverse Fourier Transform.

    Raises:
        ValueError: Lenght of x should be a power of 2.

    Returns:
        ft: FT[x] or FT^-1
    """
    N = len(x)
    # Check if N power of 2
    if N & (N-1) != 0:
        raise ValueError(
            'Length of data should be a power of 2, not {}.'.format(N)
        )
    
    x = np.array(x, dtype=np.complex)
    ft = dft(x, N, inverse)
    # if calculating inverse, multiply result with 1/N
    if inverse:
        ft *= 1/N
    return ft

def fft3d(x, inverse=False):
    """3D Fourier Transform of x using fft function.

    Args:
        x (numpy.array): 3D numpy array containg values to Fourier Transform.
        inverse (bool, optional): If True, perform inverse Fourier Transform.

    Returns:
        numpy.array: FT[x] or FT^-1[x]
    """
    ft_x = np.zeros(x.shape, dtype=np.complex)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ft_x[i,j,:,] = fft(x[i,j,:], inverse)
        for k in range(x.shape[2]):
            ft_x[i,:,k] = fft(ft_x[i, :, k], inverse)
    
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            ft_x[:,j,k] = fft(ft_x[:, j, k], inverse)    
    return ft_x

if __name__ == '__main__':

    PLOT_DIR = './plots/'

    # Generate random particles
    np.random.seed(121)

    # Grid shape
    SHAPE = 16

    positions = np.random.uniform(low=0,high=SHAPE,size=(3,1024))

    # Define grid
    grid = np.zeros((SHAPE,SHAPE,SHAPE))

    #2a
    
    # Calulate density contrast
    density_contrast = make_3D_density_contrast_grid(positions.T, grid.shape)

    # plot denisty contrast at z = 4, 9, 11 and 14
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for z, ax in zip([4,9,11,14], axes.flat):
        # plt.figure(figsize=(8,8))
        im = ax.imshow(
            density_contrast[:,:,z], 
            origin='lower', 
            extent=[0, SHAPE, 0, SHAPE],
            vmin=-1, 
            vmax=5.5
        )
        ax.set_title('z = {}'.format(z))
        ax.set_xticks(np.arange(0,16+4,4))
        ax.set_yticks(np.arange(0,16+4,4))
        ax.tick_params(length=0)
    fig.text(0.45, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')
    cb = fig.colorbar(im, ax=axes.ravel().tolist())
    cb.set_label(r'$\delta$')
    plt.savefig(os.path.join(PLOT_DIR, '2a_density_contrast.png'), dpi=200)
    plt.clf()

    #2b
    grad2phi = 1 + density_contrast
    ft_grad2phi = fft3d(grad2phi)

    # Generate k^2 grid
    x = y = z = np.arange(-(SHAPE//2-0.5), (SHAPE//2-0.5)+1, 1)
    yy, xx, zz = np.meshgrid(y, x, z, sparse=True)
    k2 = xx**2 + yy**2 + zz**2

    # Transpose k^2-grid to match coordinates of FT(nabla^2 phi)
    k2_transposed =\
        np.roll(
            np.roll(
                np.roll(
                        k2, SHAPE//2, axis=0
                    ),
                    SHAPE//2,axis=1
                ), 
            SHAPE//2, axis=2
        )
    
    # FT(phi) = FT(nabla^2 phi)/k^2
    ft_phi = ft_grad2phi/ k2_transposed

    # Transform back to real space to obtain phi
    phi = fft3d(ft_phi, inverse=True)

    # Plot phi at z = 4, 9, 11 and 14
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for z, ax in zip([4,9,11,14], axes.flat):
        im = ax.imshow(
            phi[:,:,z].real, 
            origin='lower', 
            extent=[0, SHAPE, 0, SHAPE],
            # vmin=1, 
            # vmax=1.7,
        )
        ax.set_title('z = {}'.format(z))
        ax.set_xticks(np.arange(0,16+4,4))
        ax.set_yticks(np.arange(0,16+4,4))
        ax.tick_params(length=0)
    fig.text(0.45, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')
    cb = fig.colorbar(im, ax=axes.ravel().tolist())
    cb.set_label(r'${\Phi}$')
    plt.savefig(os.path.join(PLOT_DIR, '2b_phi.png'), dpi=200)
    plt.clf()

    # Plot log10(|FT[phi]|) at z = 4, 9, 11 and 14
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for z, ax in zip([4,9,11,14], axes.flat):
        im = ax.imshow(
            np.log10(abs(ft_grad2phi[:,:,z])), 
            origin='lower', 
            extent=[0, SHAPE, 0, SHAPE],
            vmin=0.3, 
            vmax=2.4,
        )
        ax.set_title('z = {}'.format(z))
        ax.set_xticks(np.arange(0,16+4,4))
        ax.set_yticks(np.arange(0,16+4,4))
        ax.tick_params(length=0)
    fig.text(0.45, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')
    cb = fig.colorbar(im, ax=axes.ravel().tolist())
    cb.set_label(r'$\log_{10}(|\widetilde{\Phi} |)$')
    plt.savefig(os.path.join(PLOT_DIR, '2b_FT_phi.png'), dpi=200)
    plt.clf()