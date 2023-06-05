import torch
import numpy as np

def torch_polar_azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = torch.from_numpy(np.indices([H, W]))
    radius = torch.sqrt((x - H/2)**2 + (y - W/2)**2)
    radius = radius.long().flatten()
    nr = torch.bincount(radius)
#    bincount = lambda inds, arr: torch.scatter_reduce(arr, 0, inds, reduce="sum") #torch.bincount(radius, image.ravel())
    tbin = torch.bincount(radius.cuda(), image.ravel())
    radial_prof = tbin.cuda() / (nr + 1e-10).cuda()
    return radial_prof[1:] # We ignore the last two extremely high frequency bands to avoid noise.

def numpy_polar_azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = np.indices([H, W])
    radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)
    radius = radius.astype(int).ravel()
    nr = np.bincount(radius)
    tbin = np.bincount(radius, image.ravel())
    radial_prof = tbin / (nr + 1e-10)
    return radial_prof[1:] # We ignore the last two extremely high frequency bands to avoid noise.


def torch_azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = torch.from_numpy(np.indices(image.shape))

    if not center:
        center = [(x.max()-x.min())/2.0, (x.max()-x.min())/2.0]

    r = torch.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = torch.argsort(r.flatten())
    r_sorted = r.flatten()[ind]
    i_sorted = image.flatten()[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.int() #floor() # or round()?

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = torch.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = torch.cumsum(i_sorted, dim=0)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin.cuda() / nr.cuda()

    return radial_prof


def numpy_azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def per_bw(psd, p, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(psd.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = psd.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    total_energy = psd.sum()
    energy = 0
    for i, pd in enumerate(tbin):
        energy += pd
        if energy >= p * total_energy:
          return i 
    return None
