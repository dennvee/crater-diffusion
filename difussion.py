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


class Crater:
    def __init__(self, name, center=None, box_size=None, demfile=None, printsize=100, wallsize=4, mindepth=20, res=512, skips=1, issynthetic=False, diameter=None, show=True, extent_radius_ratio=3, debugshow=None):
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
        self.create_mesh()
        self.mesh2mold()
        if show:
            self.show()

    def read_nac_dtm(self):
        filename = self.demfile
        center = self.center
        box_size = self.box_size
        res = self.res
        read_external_data = filename is None
        isboundary = False
        self.gridshape = (0,0)
         
        if res < 256 and center[0] > 180:
            center = (center[0]-360,center[1])
        
        def _rotate_coords(X, Y, Z, center):
            # Convert crater's longitude and latitude to radians
            lon0 = np.radians(center[0])
            lat0 = np.radians(center[1])

            # Rotation around Z-axis to align longitude
            rotation_z = R.from_rotvec(-lon0 * np.array([0, 0, 1]))
            
            # Rotation around Y-axis to align latitude and point north along X-axis
            rotation_y = R.from_rotvec((lat0 - np.pi / 2) * np.array([0, 1, 0]))
            
            # Combine rotations
            rotation = rotation_y * rotation_z  # Apply rotation_z first, then rotation_y

            # Stack X, Y, Z into an array of shape (N, 3)
            coords = np.vstack((X, Y, Z)).T  # Shape: (N, 3)
            
            # Apply the rotation to all coordinates
            rotated_coords = rotation.apply(coords)
            
            # Extract rotated coordinates
            X_rot = rotated_coords[:, 0]
            Y_rot = rotated_coords[:, 1]
            Z_rot = rotated_coords[:, 2]
            return X_rot, Y_rot, Z_rot
                
        def _get_sldem_file(center,res,boundary_offset=(0,0)):
            valid_res = [4,16,64,128,256,512,1024]
            
            if res not in valid_res:
                raise ValueError("Invalid resolution.")
            if np.abs(center[1]) < 60 and (res == 256 or res == 512):
                use_sldem = True
            else:
                use_sldem = False 
            
            if use_sldem:
                src_url = f"https://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/"
                
                if center[1] > 0:
                    latdir = 'N'
                else:
                    latdir = 'S'     
                    
                if res == 512:
                    dlon = 45
                    dlat = 30
                elif res == 256:
                    dlon = 120
                    dlat = 60 
            else:
                src_url = f"https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/jp2/"
                if center[1] > 0:
                    latdir = 'n'
                else:
                    latdir = 's'                      
                if res == 1024:
                    dlon = 30
                    dlat = 15
                elif res == 512:
                    dlon = 90
                    dlat = 45
                elif res == 256:
                    dlon = 180
                    dlat = 90
         
            if res >= 256: 
                lonval = np.abs(center[0] / dlon) + boundary_offset[0]
                lonlo = int(np.floor(lonval) * dlon) % 360
                lonhi = int(lonlo + dlon) 
            
                latval = np.abs(center[1] / dlat)  + boundary_offset[1]
                latlo = int(np.floor(latval) * dlat)
                lathi = int(latlo + dlat)
                
                if latlo < 0:
                    if latdir.upper() == 'N':
                        latdir = 'S'
                    else:
                        latdir = 'N'
                    latlo = -latlo
                    
                if lathi < 0:
                    if latdir.upper() == 'N':
                        latdir = 'S'
                    else:
                        latdir = 'N'
                    lathi = -lathi
                    
                if latdir.upper() == 'S':
                    tmp = latlo
                    latlo = lathi
                    lathi = tmp
                    
                if lathi > 90:
                    lathi = 90
                if latlo >= 90:
                    latlo = latlo - dlat
                
            if use_sldem:
                if res == 512:
                    filename = f"SLDEM2015_{res:3d}_{latlo:02d}{latdir.upper()}_{lathi:02d}{latdir.upper()}_{lonlo:03d}_{lonhi:03d}.JP2"
                elif res == 256:
                    filename = f"SLDEM2015_{res:3d}_{latlo:d}{latdir.upper()}_{lathi:d}{latdir.upper()}_{lonlo:03d}_{lonhi:03d}.JP2"
            else:
                if res >= 256:
                    filename = f"ldem_{res:d}_{latlo:02d}{latdir.lower()}_{lathi:02d}{latdir.lower()}_{lonlo:03d}_{lonhi:03d}.jp2"
                else:
                    filename = f"ldem_{res:d}.jp2"
                
            filename_full = os.path.join(self.demdir,filename)
            if not os.path.exists(filename_full):
                url = f"{src_url}{filename}"
                print(f"Downloading {url} to {filename_full}")
                download(url,filename_full)
            return filename
        
        def _get_dem(filename=None, filelist=None, compute_boundary=True):
            isboundary = False
            boundary = [None,None]
            
            with MemoryFile() as memfile:
                if filelist is not None:
                    print(f"Merging files: {filelist}")
                    src_list = []
                    for filename in filelist:
                        src = rasterio.open(os.path.join(self.demdir,filename))
                        src_list.append(src)
                    mosaic, transform = merge(src_list) 
                    out_meta = src.meta.copy()
                    out_meta.update({
                            "driver": "GTiff",
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": transform,
                            "crs": src.crs
                        }
                    )
                    with memfile.open(**out_meta) as dst:
                        dst.write(mosaic)
                        
                    cm = memfile.open()
                else:
                    cm = rasterio.open(os.path.join(self.demdir,filename)) 
                with cm as src:
                    # Desired box size in meters
                    self.radius = src.crs.data['R']
                    half_box_size = self.expansion_buffer * box_size / 2 # Make the box slightly larger than the desired size, as it will be truncated into a square of the correct size later
                    with WarpedVRT(src, crs=dst_crs, resampling=Resampling.cubic) as vrt:
                        
                        d = self.radius * np.asin(half_box_size / self.radius) 
                        # Compute the window in the destination CRS
                        x_min = -d
                        x_max = d
                        y_min = -d
                        y_max = d

                        # Compute the window in pixel coordinates
                        window_orig = from_bounds(x_min, y_min, x_max, y_max, vrt.transform)
                        
                        if compute_boundary: 
                            window = window_orig.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                            if window != window_orig:
                                print("Warning: Window was adjusted to fit within the raster bounds.")
                                if read_external_data:
                                    isboundary = True
                                    if window.col_off != window_orig.col_off:
                                        boundary[0] = 'W'
                                    elif window.width != window_orig.width:
                                        boundary[0] = 'E'                            
                                    if window.row_off != window_orig.row_off:
                                        boundary[1] = 'N'
                                    elif window.height != window_orig.height:
                                        boundary[1] = 'S'
                                return None, isboundary, boundary
                            
                        else:
                            window = window_orig

                        # Read the data within the window
                        elevation = vrt.read(1, window=window)
                        
                        # Handle nodata values
                        mask_nodata = elevation != vrt.nodata
                        mean_elevation = np.mean(elevation[mask_nodata])
                        elevation[~mask_nodata] = mean_elevation 
                        elevation = elevation.astype(np.float32) 

                        # Compute the transform for the window
                        window_transform = vrt.window_transform(window)
                       
                        self.gridshape = elevation.shape
                        print(f"Grid shape: {self.gridshape}")
                         
                        # Generate row and column indices
                        rows, cols = np.indices(elevation.shape)

                        # Compute the x and y coordinates of each pixel in the local CRS
                        x_coords, y_coords = rasterio.transform.xy(
                            window_transform, rows, cols, offset='center'
                        )
                        x_coords = np.array(x_coords)
                        y_coords = np.array(y_coords)

                elevation_flat = elevation.flatten()
                
                # Transformer from local CRS to geodetic CRS
                geodetic_crs = CRS.from_proj4("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
                transformer_to_geodetic = Transformer.from_crs(dst_crs, geodetic_crs, always_xy=True)

                # Transform to longitude and latitude
                longitudes, latitudes = transformer_to_geodetic.transform(x_coords, y_coords)
                longitudes, latitudes = np.radians(longitudes), np.radians(latitudes)
                r_vals = self.radius + elevation_flat
                
                X = r_vals * np.cos(latitudes) * np.cos(longitudes)
                Y = r_vals * np.cos(latitudes) * np.sin(longitudes)
                Z = r_vals * np.sin(latitudes)          
                
                X,Y,Z = _rotate_coords(X,Y,Z,center)  
                
                # Create the DEM dataset
                dem = xr.Dataset(
                    data_vars={'elevation': (["loc"], elevation.flat)},
                    coords={
                        'X': (["loc"], X),
                        'Y': (["loc"], Y),
                        'Z': (["loc"], Z),
                        'xproj': (["loc"], x_coords),
                        'yproj': (["loc"], y_coords),
                    }
                )
                return dem, isboundary, boundary
            
        
        dst_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center[1]} +lon_0={center[0]} "
            "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
        )         
            
        if read_external_data: 
            compute_boundary = res >= 256
            filename = _get_sldem_file(center,res)
        else:
            compute_boundary = False
        
        dem, isboundary, boundary = _get_dem(filename, compute_boundary=compute_boundary)
        if isboundary:
            filelist = [filename]
            boundary_offset=[0,0]
            if boundary[0] is not None:
                if boundary[0] == 'W':
                    boundary_offset = (-1,0)
                elif boundary[0] == 'E':
                    boundary_offset = (1,0)
                filelist.append(_get_sldem_file(center,res,boundary_offset))
            if boundary[1] is not None:
                if boundary[1] == 'N':
                    boundary_offset = (0,int(np.sign(center[1])))
                elif boundary[1] == 'S':
                    boundary_offset = (0,-int(np.sign(center[1])))
                filelist.append(_get_sldem_file(center,res,boundary_offset))
            dem, _, _ = _get_dem(filelist=filelist)
           
        self.dem = dem
        
        self.pix = (
            (self.dem.X.max() - self.dem.X.min()).values.item() / self.gridshape[0], 
            (self.dem.Y.max() - self.dem.Y.min()).values.item() / self.gridshape[1]
        )             
        return 

    
    def create_synthetic_crater(self):
        if self.diameter is None:
            raise ValueError("Diameter must be specified for synthetic craters.")
        dkm = self.diameter*1e-3
        demfile = os.path.join(os.getcwd(),"synthetic_craters","dem",f"D_{dkm:.2f}_km_z.npy")  
        if not os.path.exists(demfile):
            sc.create_synthetic_craters([dkm]) 
        x_coords = np.load(os.path.join(os.getcwd(),"synthetic_craters","dem",f"D_{dkm:.2f}_km_x.npy"))
        y_coords = np.load(os.path.join(os.getcwd(),"synthetic_craters","dem",f"D_{dkm:.2f}_km_y.npy"))
        demarray = np.load(demfile)
        self.gridshape = demarray.shape
        self.dem = xr.Dataset(
            data_vars={'elevation': (["loc"], demarray.flatten())},
            coords={
                'X': (["loc"], x_coords.flatten()), # Compute this properly
                'Y': (["loc"], y_coords.flatten()),
                'R': (["loc"], np.sqrt(x_coords**2 + y_coords**2).flatten()),
                'xproj': (["loc"], x_coords.flatten()),
                'yproj': (["loc"], y_coords.flatten()),
            },
        )
        self.radius = 1737.4e3
        dZ = self.dem['R']**2 / self.radius
        self.dem['Z'] = self.dem['elevation']-dZ
        self.dem['dZ'] = -dZ
        self.skips=1
        return

    def create_mesh(self):
        
        # Scale the DEM to the desired print size
        printsize = self.printsize
        
        extent = self.dem['X'].max() - self.dem['X'].min(), self.dem['Y'].max() - self.dem['Y'].min()  
        
        scalefac = self.expansion_buffer * printsize / min(extent)
        ds = self.dem.copy()
        ds['X'] *= scalefac
        ds['Y'] *= scalefac
        ds['Z'] *= scalefac
        ds['elevation'] *= scalefac
        self.radius *= scalefac
        ds['Z'] -= self.radius

        # Shift Z values so the lowest point is at zero
        min_z = ds['Z'].min()
        ds['Z'] -= min_z        
        
        wallsize=self.wallsize
        skips=self.skips
        mindepth=self.mindepth
        
        if skips > 1:
            print(f"Skipping every {skips} points in the grid for the mesh.")

        # Reshape the data into 2D arrays
        X = -ds['X'].values.reshape(self.gridshape)[::skips, ::skips]
        Y = ds['Y'].values.reshape(self.gridshape)[::skips, ::skips]
        Z = -ds['Z'].values.reshape(self.gridshape)[::skips, ::skips]

        gridshape = X.shape  # (number_of_rows, number_of_columns)
        if skips > 1:
            print(f"New mesh resolution is {gridshape}")

        # Create vertices
        vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        # Create faces using correct index mapping
        faces = []
        for i in range(gridshape[0] - 1):
            for j in range(gridshape[1] - 1):
                idx = i * gridshape[1] + j
                faces.append([idx + gridshape[1], idx + 1, idx])
                faces.append([idx + gridshape[1], idx + gridshape[1] + 1, idx + 1])

        faces = np.array(faces)

        # Create the top surface mesh
        top_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if self.debugshow == "top":
            scene = trimesh.Scene()
            scene.add_geometry([top_mesh])
            scene.show() 
        
        # Create the bottom surface mesh (flat plane at base height)
        height = Z.max() - Z.min() + wallsize
        if height > mindepth:
            depth = wallsize
        else:
            depth = mindepth - height 
        bottom_z = np.full_like(Z, depth)
        bottom_vertices = np.column_stack((X.flatten(), Y.flatten(), bottom_z.flatten()))
        bottom_faces = faces.copy()
        bottom_faces = np.fliplr(bottom_faces)  # Reverse the face orientation

        bottom_mesh = trimesh.Trimesh(vertices=bottom_vertices, faces=bottom_faces)

        # Create side walls
        side_faces = []
        side_vertices = []

        def add_side(v0, v1, v2, v3, reverse=False):
            idx = len(side_vertices)
            side_vertices.extend([v0, v1, v2, v3])
            if not reverse:
                side_faces.append([idx, idx + 1, idx + 2])
                side_faces.append([idx, idx + 2, idx + 3])
            else:
                side_faces.append([idx, idx + 2, idx + 1])
                side_faces.append([idx, idx + 3, idx + 2])

        # Reshape vertices to 2D grid for edge access
        top_vertices_grid = vertices.reshape((gridshape[0], gridshape[1], 3))
        bottom_vertices_grid = bottom_vertices.reshape((gridshape[0], gridshape[1], 3))

        # Front wall (Y is minimum)
        for i in range(gridshape[0] - 1):
            v0 = top_vertices_grid[i, 0]
            v1 = top_vertices_grid[i + 1, 0]
            v2 = bottom_vertices_grid[i + 1, 0]
            v3 = bottom_vertices_grid[i, 0]
            add_side(v0, v1, v2, v3, reverse=True)

        # Back wall (Y is maximum)
        for i in range(gridshape[0] - 1):
            v0 = top_vertices_grid[i, -1]
            v1 = top_vertices_grid[i + 1, -1]
            v2 = bottom_vertices_grid[i + 1, -1]
            v3 = bottom_vertices_grid[i, -1]
            add_side(v0, v1, v2, v3, reverse=False)

        # Left wall (X is minimum)
        for j in range(gridshape[1] - 1):
            v0 = top_vertices_grid[0, j]
            v1 = top_vertices_grid[0, j + 1]
            v2 = bottom_vertices_grid[0, j + 1]
            v3 = bottom_vertices_grid[0, j]
            add_side(v0, v1, v2, v3, reverse=False)

        # Right wall (X is maximum)
        for j in range(gridshape[1] - 1):
            v0 = top_vertices_grid[-1, j]
            v1 = top_vertices_grid[-1, j + 1]
            v2 = bottom_vertices_grid[-1, j + 1]
            v3 = bottom_vertices_grid[-1, j]
            add_side(v0, v1, v2, v3, reverse=True)

        # Create the side walls mesh
        side_mesh = trimesh.Trimesh(vertices=np.array(side_vertices), faces=np.array(side_faces))
        
        # Combine all meshes
        self.mesh = trimesh.util.concatenate([top_mesh, bottom_mesh, side_mesh])
        
        # Ensure that self.mesh is watertight for boolean operations
        if not self.mesh.is_watertight:
            print("Mesh is not watertight. Attempting to repair.")
            self.mesh.fill_holes()
            self.mesh.remove_unreferenced_vertices()
            self.mesh.merge_vertices()
            self.mesh.fix_normals()
            
        # Verify if the mesh is now a volume
        if self.mesh.is_volume:
            print("Mesh is a volume.")
        else:
            print("Mesh is still not a volume.")            
           
        if self.debugshow == "mesh": 
            scene = trimesh.Scene()
            scene.add_geometry([self.mesh])
            scene.show() 
        


        # Export the mesh to an STL file
        self.mesh.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
        self.mesh.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0]))
        self._square_off_mesh()
        
        output_filename=os.path.join(self.stldir,f"{self.name.lower()}_crater.stl")
        self.mesh.export(output_filename)
        print(f"Mesh exported to {output_filename}")
    
    def _square_off_mesh(self):
        printsize = self.printsize
        # Truncates the mesh to a square box of the specified size
        
        # Get the bounding box of the mesh
        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1] 
        
        box_buffer = 1.1
               
        height = box_buffer*max(np.abs(bbox_min[1]),np.abs(bbox_max[1]))
        outer_min = [box_buffer*bbox_min[0],-height,box_buffer*bbox_min[2]]
        outer_max = [box_buffer*bbox_max[0],height,box_buffer*bbox_max[2]]
         
        inner_box = trimesh.creation.box(bounds=[[-printsize/2,-height,-printsize/2],[printsize/2,height,printsize/2]])
        outer_box = trimesh.creation.box(bounds=[outer_min,outer_max])
        box_mesh = trimesh.boolean.difference([outer_box, inner_box], engine='manifold')
      
        if self.debugshow == "box": 
            scene = trimesh.Scene()
            box_mesh.visual.vertex_colors = np.array([0,0,255],dtype=np.uint8)
            scene.add_geometry([self.mesh,box_mesh])
            scene.show() 
            
        if self.debugshow == "boxintersect":
            intersection = trimesh.boolean.intersection([self.mesh, box_mesh], engine='manifold')
            intersection.visual.vertex_colors = np.array([255,0,0],dtype=np.uint8)
            scene = trimesh.Scene()
            scene.add_geometry([self.mesh,intersection])
            scene.show()
            
        self.mesh = trimesh.boolean.difference([self.mesh, box_mesh], engine='manifold')
        
        # Ensure that self.mesh is watertight for boolean operations
        if not self.mesh.is_watertight:
            print("Mesh is not watertight. Attempting to repair.")
            self.mesh.fill_holes()
            self.mesh.remove_unreferenced_vertices()
            self.mesh.merge_vertices()
            self.mesh.fix_normals()        
        
        if self.debugshow == "squared": 
            scene = trimesh.Scene()
            scene.add_geometry([self.mesh])
            scene.show() 
        
        return 
        

    def apply_diffusion(self, diffusion_coefficient, time_step, total_time):
        """
        Apply topographic diffusion to DEM using an finite-difference method.

        Args:
            diffusion_coefficient (float): Diffusion coefficient (k) in m^2/s.
            time_step (float): Time step for the simulation (dt).
            total_time (float): Total simulation time.
        """
        print(f"Applying topographic diffusion: k={diffusion_coefficient}, dt={time_step}, T={total_time}")
        
        # Extract elevation data from the DEM
        elevation = self.dem['elevation'].values.reshape(self.gridshape)
        dx, dy = self.pix  # Grid spacing in x and y directions

        # Calculate number of time steps
        num_steps = int(total_time / time_step)

        for step in range(num_steps):
            # Compute gradients (central differences)
            grad_x = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / (2 * dx)
            grad_y = (np.roll(elevation, -1, axis=0) - np.roll(elevation, 1, axis=0)) / (2 * dy)

            # Compute divergence (central differences)
            div_x = (np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1)) / (2 * dx)
            div_y = (np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0)) / (2 * dy)

            # Update elevation using the diffusion equation
            elevation += time_step * diffusion_coefficient * (div_x + div_y)

            # Print progress
            if step % (num_steps // 10) == 0:
                print(f"Step {step}/{num_steps} complete")

        # Update DEM with diffused elevation
        self.dem['elevation'].values = elevation.flatten()
        print("Topographic diffusion applied.")

    def apply_topographic_diffusion(self):
        """
        Wrapper to apply diffusion based on available DEM data.
        """
        if self.dem is None:
            raise ValueError("DEM data is not available for diffusion.")
        
       # Diffusion parameters
        diffusion_coefficient = 0.1  # m^2/s
        time_step = 500000  # seconds
        total_time = 3.16e13  # 1 million years in seconds

        # Apply the diffusion process
        self.apply_diffusion(diffusion_coefficient, time_step, total_time)


    def mesh2mold(self):
        wallsize=self.wallsize
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet.")

        # Get the bounding box of the mesh
        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1]

        # Adjust the bounding box to create the mold box
        box_min = bbox_min - [0, 0, 0]  
        box_max = bbox_max + [0, wallsize, 0]

        # Create the box mesh
        box_mesh = trimesh.creation.box(bounds=[box_min, box_max])

        if self.debugshow == "moldbox": 
            scene = trimesh.Scene()
            box_mesh.visual.vertex_colors = trimesh.visual.random_color()
            scene.add_geometry([box_mesh,self.mesh])
            scene.show()    

        # Perform the boolean difference using the 'manifold' engine
        print("Performing boolean subtraction...")
        mold_mesh = trimesh.boolean.difference([box_mesh, self.mesh], engine='manifold')
        
        # Extend bottom of mold to create anchor points for the remaining walls
        anchor_bounds_lo = np.array([box_min[0] - 4*wallsize, box_min[1], box_min[2]-wallsize]) 
        anchor_bounds_hi = np.array([box_max[0] + 4*wallsize, box_max[1], box_min[2]]) 
        anchor_mesh = trimesh.creation.box(bounds=[anchor_bounds_lo,anchor_bounds_hi])
        mold_mesh = trimesh.boolean.union([mold_mesh,anchor_mesh],engine='manifold')
        
        if mold_mesh is None or len(mold_mesh.vertices) == 0:
            raise ValueError("Boolean operation failed. The resulting mesh is empty.")
        
        screw_hole_radius = 1.0
        screw_hole_height = 2*wallsize
        lshift = trimesh.transformations.translation_matrix([anchor_bounds_lo[0] + screw_hole_height, anchor_bounds_hi[1]-screw_hole_height, anchor_bounds_lo[2]])
        rshift = trimesh.transformations.translation_matrix([anchor_bounds_hi[0] - screw_hole_height, anchor_bounds_hi[1]-screw_hole_height, anchor_bounds_lo[2]])
        lscrew = trimesh.creation.cylinder(radius=screw_hole_radius, height=3*wallsize, transform=lshift)
        rscrew = trimesh.creation.cylinder(radius=screw_hole_radius, height=3*wallsize, transform=rshift)
        
        mold_mesh = trimesh.boolean.difference([mold_mesh, lscrew, rscrew], engine='manifold')
        
        if self.debugshow == "mold": 
            scene = trimesh.Scene()
            scene.add_geometry([mold_mesh])
            scene.show()         

        # Store the mold mesh
        self.mold = mold_mesh
        
        # Export the mold mesh to an STL file
        output_filename = os.path.join(self.stldir,f"{self.name.lower()}_crater_mold.stl")
        self.mold.export(output_filename)
        print(f"Mold mesh exported to {output_filename}")
   
    def show(self):
        rotation_180_y = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        rotation_90_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        
        # Define the spacing between the meshes
        spacing = 10  # Adjust the spacing as needed
        scene = trimesh.Scene()
        
        # Create copies of the meshes to avoid modifying the originals
        if self.mesh is not None:
            mesh_transformed = self.mesh.copy()
            mesh_transformed.apply_transform(rotation_90_x)
            scene.add_geometry(mesh_transformed)
        if self.mold is not None:
            mold_transformed = self.mold.copy()
            mold_transformed.apply_transform(rotation_180_y)
            mold_bounds = mold_transformed.bounds
            mold_size_x = mold_bounds[1][0] - mold_bounds[0][0]
            translate_mold_x = (mold_size_x + spacing)
            translation_mold = trimesh.transformations.translation_matrix([translate_mold_x, 0, 0])
            mold_transformed.apply_transform(translation_mold)
            mold_transformed.apply_transform(rotation_90_x)
            scene.add_geometry(mold_transformed)

        # Show the scene
        scene.show()

    def save_mesh(self, mesh, filename):
        """
        Save a given mesh to an STL file.

        Args:
            mesh (trimesh.Trimesh): The mesh to save.
            filename (str): The filename to save the mesh as.
        """
        if not mesh:
            raise ValueError("Mesh is empty or not defined.")
        output_path = os.path.join(self.stldir, filename)
        mesh.export(output_path)
        print(f"Mesh saved to {output_path}")

    def save_before_and_after_diffusion(self, before_filename, after_filename):
        """
        Save the mesh before and after applying topographic diffusion.

        Args:
            before_filename (str): Filename to save the mesh before diffusion.
            after_filename (str): Filename to save the mesh after diffusion.
        """
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet.")
        
        # Save the mesh before diffusion
        print("Saving 'before diffusion' mesh...")
        self.save_mesh(self.mesh, before_filename)

        # Apply diffusion
        self.apply_topographic_diffusion()

        # Save the mesh after diffusion
        print("Saving 'after diffusion' mesh...")
        self.save_mesh(self.mesh, after_filename)
     
crater_catalog = {
    'Linne': {
        'center': (11.7994, 27.7464), 
        'box_size': 5.0e3, 
        'demfile': 'NAC_DTM_LINNECRATER.TIF',
        'skips' : 1
    },
    'Flamsteed': {
        'center': (315.6602, -4.4906), 
        'diameter': 19.34e3, 
        'res' : 512,
        'skips' : 1
    },
    'Landsberg B': {
        'center': (331.8606, -2.4939),
        'diameter': 9.0e3, 
        'res' : 512,
        'skips' : 1
    },                  
    'Luther' : {
        'center' : (24.1502, 33.1967),
        'diameter' : 9.29e3,
        'res' : 512,
        'skips' : 1
    },
    'Copernicus' : {
        'center' : (339.9214, 9.6209),
        'diameter' : 96.07e3,
        'res' : 512,
        'skips' : 3
    },
    'Kepler' : {
        'center' : (321.9913, 8.121),
        'diameter' : 29.49e3,
        'res' : 512,
        'skips' : 1
    },
    'Aristarchus' : {
        'center' : (312.5099, 23.7299),
        'diameter' : 40e3,
        'res' : 512,
        'skips' : 1
    },        
    'Pythagoras' : {
        'center' : (297.0241, 63.6814),
        'diameter' : 144.6e3,
        'res' : 256,
        'skips' : 4
    }, 
    'Schrodinger' : {
        'center' : (132.9252, -74.7326),
        'diameter' : 316.4e3,
        'res' : 16,
        'skips' :1 
    },
    'Orientale' : {
        'center' : (261.3, -19.8),
        'diameter' : 930e3,
        'extent_radius_ratio' : 1.5,
        'res' : 16,
        'skips' : 1
    },
    "Du_2.4km": {
        'diameter' : 2.4e3,
        'issynthetic' : True
    },
    "Du_19.53km": {
        'diameter' : 19.53e3,
        'issynthetic' : True
    },
        "Du_94.80km": {
        'diameter' : 94.80e3,
        'issynthetic' : True
    }                                        
}     




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
        crater = Crater(name=name, **crater_catalog[name], show=show, debugshow=debugshow)
