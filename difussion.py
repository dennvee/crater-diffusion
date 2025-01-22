import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from rasterio.io import MemoryFile
from scipy.spatial.transform import Rotation as R
from pyproj import CRS, Transformer
import xarray as xr
import trimesh
from trimesh.transformations import rotation_matrix
import os
import requests
import sys
from synthetic_craters import utilities as sc
import os
from tqdm import tqdm

def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def apply_topographic_diffusion(dem, diffusion_coefficient, time_steps, dt):
    """
    Apply topographic diffusion to the elevation data.
    Parameters:
    - dem: xarray.Dataset, the DEM dataset with 'elevation'.
    - diffusion_coefficient: float, the diffusion coefficient (κ).
    - time_steps: int, the number of time steps for diffusion.
    - dt: float, the time step size.
    Returns:
    - diffused_dem: xarray.Dataset, the updated DEM dataset.
    """
    elevation = dem['elevation'].values.reshape((dem.dims['loc'],))
    dx = dem['xproj'][1] - dem['xproj'][0]  # Assuming uniform spacing
    dy = dem['yproj'][1] - dem['yproj'][0]

    for _ in range(time_steps):
        # Compute the Laplacian
        laplacian = (
            (np.roll(elevation, 1, axis=0) - 2 * elevation + np.roll(elevation, -1, axis=0)) / dx**2 +
            (np.roll(elevation, 1, axis=1) - 2 * elevation + np.roll(elevation, -1, axis=1)) / dy**2
        )
        # Update elevation
        elevation += diffusion_coefficient * laplacian * dt

    # Update the DEM dataset
    diffused_dem = dem.copy()
    diffused_dem['elevation'] = ('loc', elevation.flatten())
    return diffused_dem


class Crater:
    def __init__(self, name, center=None, box_size=None, demfile=None, printsize=100, wallsize=4, mindepth=20, res=512, skips=1, issynthetic=False, diameter=None, show=True, extent_radius_ratio=3, debugshow=None, apply_diffusion=False, diffusion_params=None):
        self.demdir = os.path.join(os.getcwd(),'data') 
        if not os.path.exists(self.demdir):
            os.makedirs(self.demdir)
        self.stldir = os.path.join(os.getcwd(),'stl')
        if not os.path.exists(self.stldir):
            os.makedirs(self.stldir) 
        self.dem = None
        self.pix = None
        self.gridshape = None
        self.mesh = None
        self.mold = None
        self.container = None
        self.name = name
        self.printsize=printsize
        self.wallsize=wallsize
        self.mindepth=mindepth
        self.debugshow=debugshow
        self.apply_diffusion = apply_diffusion
        self.diffusion_params = diffusion_params or {'diffusion_coefficient': 0.01, 'time_steps': 100, 'dt': 1.0}

        if issynthetic:
            self.expansion_buffer = 1.0
            self.diameter = diameter
            self.create_synthetic_crater()
        else:
            self.expansion_buffer = 1.05
            self.res=res
            self.skips=skips
            self.demfile=demfile
            self.center = center
            if box_size is None:
                if diameter is None:
                    raise ValueError("Either box_size or diameter must be specified.")
                self.box_size = diameter * extent_radius_ratio
            else:
                self.box_size = box_size
            self.read_nac_dtm()

        # Apply diffusion if specified
        if self.apply_diffusion:
            self.apply_topographic_diffusion()

        self.create_mesh()
        self.mesh2mold()
        if show:
            self.show()

    def apply_topographic_diffusion(self):
        """
        Apply topographic diffusion to the DEM.
        """
        if self.dem is None:
            raise ValueError("DEM data is not available for diffusion.")
        
        print("Applying topographic diffusion...")
        params = self.diffusion_params
        self.dem = apply_topographic_diffusion(self.dem, params['diffusion_coefficient'], params['time_steps'], params['dt'])
        print("Topographic diffusion applied.")

    def save_dem(self, filename):
        """
        Save the DEM data to a .npy file.
        """
        if self.dem is None:
            raise ValueError("DEM data is not available.")
        
        elevation = self.dem['elevation'].values.reshape(self.gridshape)
        np.save(filename, elevation)
        print(f"DEM saved to {filename}.")

    def save_mesh(self, filename):
        """
        Save the mesh as an STL file.
        """
        if self.mesh is None:
            raise ValueError("Mesh is not available.")
        
        self.mesh.export(filename)
        print(f"Mesh saved to {filename}.")

    # Other methods (read_nac_dtm, create_mesh, create_synthetic_crater, etc.) remain unchanged.

if __name__ == '__main__':
    debugshow = None 
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <crater_name> [optional debugshow value: top|mesh|squared|box|moldbox|boxintersect|mold]")
        namelist = list(crater_catalog.keys())
        show = False
        namelist = ["Copernicus"]
        show = True
    else:
        namelist = [sys.argv[1]]
        show = True
        if len(sys.argv) == 3:
            debugshow = sys.argv[2]
        
    for name in namelist: 
        crater = Crater(
            name=name,
            **crater_catalog[name],
            show=show,
            debugshow=debugshow,
            apply_diffusion=True,  # Set to True to apply diffusion
            diffusion_params={'diffusion_coefficient': 0.01, 'time_steps': 500, 'dt': 0.1}  # Example params
        )
        # Save before and after DEM
        crater.save_dem(f"{name}_initial_dem.npy")
        crater.apply_topographic_diffusion()
        crater.save_dem(f"{name}_diffused_dem.npy")

        # Save meshes
        crater.save_mesh(f"{name}_initial_mesh.stl")
