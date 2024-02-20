
import numpy as np
import os
import floodsformer.utils.logg as logg

logger = logg.get_logger(__name__)

class Preprocessing():
    def __init__(
            self, 
            filename, 
            max_depth, 
            zero_depth=1e-5,
        ):
        ''' 
        Class for load, write and normalize the input maps.
        Recommended file format: Surfer binary v6 grid. More information: 
        https://surferhelp.goldensoftware.com/topics/surfer_6_grid_file_format.htm?Highlight=Surfer%206%20Grid%20File%20Format.

        Args: 
            filename (string): path to the file to read
            max_depth (float): expected maximum water depth for all the simulations (used to normalize the input data).
            zero_depth (float): threshold (m) to consider a cell dry. During the map loading
                                cells with a value lower than the threshold are set equal to 0.
        '''
        self.max_depth = max_depth  # maximum water depth of the dataset (used for normalization)
        self.zero_depth = zero_depth

        f = open (filename, 'r', errors="ignore")
        name = f.read(4)

        if name == "DSAA":    # Surfer ascii file format -> more information: https://surferhelp.goldensoftware.com/topics/ascii_grid_file_format.htm
            f.readline()
            n_cells = np.short(f.readline().replace("\n", "").split())  # number of cells (X and Y directions)
            xlim = np.double(f.readline().replace("\n", "").split())    # minimum and maximum extension in X direction
            ylim = np.double(f.readline().replace("\n", "").split())    # minimum and maximum extension in Y direction
            f.readline()
            data = f.readlines()
            data = np.asarray([l.replace("\n", "").split() for l in data]).astype(np.float32)

            # Size of the cells (equal in X and Y direction)
            c_size = round((xlim[1] - xlim[0]) / (n_cells[0] - 1), 8)

            # Save map information
            self.nx = np.short(round(n_cells[0], 8))  # number of grid lines along the X axis (columns)
            self.ny = np.short(round(n_cells[1], 8))  # number of grid lines along the Y axis (rows)
            self.xlo = np.double(round(xlim[0], 8))   # minimum X value of the grid
            self.xhi = np.double(round(xlim[1], 8))   # maximum X value of the grid
            self.ylo = np.double(round(ylim[0], 8))   # minimum Y value of the grid
            self.yhi = np.double(round(ylim[1], 8))   # maximum Y value of the grid

            self.ncol = self.nx
            self.nrow = self.ny

        elif name == "DSBB":    # file Surfer binary v6 (recommended) -> more information: https://surferhelp.goldensoftware.com/topics/surfer_6_grid_file_format.htm?Highlight=Surfer%206%20Grid%20File%20Format
            f.close()
            f = open (filename, 'rb')  # open file in binary mode
        
            n_cells = np.fromfile(f, dtype=np.short, count=2, offset=4)  # number of cells (X and Y directions) (2*2 bytes - short) -> skip 4 bytes that represent the word "DSBB" (4*1 bytes - char)
        
            # Map extensions
            xlim = np.fromfile(f, dtype=np.double, count=2)       # minimum and maximum extension in X direction (2*8 bytes - double)
            ylim = np.fromfile(f, dtype=np.double, count=2)       # minimum and maximum extension in Y direction (2*8 bytes - double)
            #zlim = np.fromfile(f, dtype=np.double, count=2)       # minimum and maximum extension in Z direction (2*8 bytes - double)

            c_size = round((xlim[1] - xlim[0]) / (n_cells[0] - 1), 8)  # Size of the cells (equal in X and Y direction)

            f.seek(56, 0)
            # read the matrix with dimensions: n_cells[1] x n_cells[0] of float32
            data = np.reshape(np.fromfile(f, dtype=np.float32), (n_cells[1], n_cells[0]))

            # Save map information
            self.nx = np.short(round(n_cells[0], 8)).tobytes()  # number of grid lines along the X axis (columns)
            self.ny = np.short(round(n_cells[1], 8)).tobytes()  # number of grid lines along the Y axis (rows)
            self.xlo = np.double(round(xlim[0], 8)).tobytes()   # minimum X value of the grid
            self.xhi = np.double(round(xlim[1], 8)).tobytes()   # maximum X value of the grid
            self.ylo = np.double(round(ylim[0], 8)).tobytes()   # minimum Y value of the grid
            self.yhi = np.double(round(ylim[1], 8)).tobytes()   # maximum Y value of the grid

            self.ncol = np.short(round(n_cells[0], 8))
            self.nrow = np.short(round(n_cells[1], 8))

        elif name == "DSRB":    # file Surfer binary v7 -> more information: https://surferhelp.goldensoftware.com/topics/surfer_7_grid_file_format.htm
            f.close()
            f = open (filename, 'rb')  # open file in binary mode

            n_cells = np.fromfile(f, dtype=np.int32, count=2, offset=20)     # number of cells (Y and X directions) => skip 20 bytes (5 long of 4 bytes)

            xLL = np.fromfile(f, dtype=np.double, count=1)  # X coordinate of the lower left corner of the grid (double)
            yLL = np.fromfile(f, dtype=np.double, count=1)  # Y coordinate of the lower left corner of the grid (double)
                
            c_size = np.fromfile(f, dtype=np.double, count=2)    # size of the cell (in X direction, the same for Y)

            xlim = np.concatenate((xLL, xLL + (n_cells[1] - 1) * c_size[0]))
            ylim = np.concatenate((yLL, yLL + (n_cells[0] - 1) * c_size[1]))
            c_size = round(c_size[0], 8)

            f.seek(100, 0)
            # read the matrix with dimensions: n_cells[0] x n_cells[1] of double
            data = np.reshape(np.fromfile(f, dtype=np.double), (n_cells[0], n_cells[1])).astype(np.float32)

            # Save map information
            self.nx = np.short(round(n_cells[1], 8)).tobytes()  # number of grid lines along the X axis (columns)
            self.ny = np.short(round(n_cells[0], 8)).tobytes()  # number of grid lines along the Y axis (rows)
            self.xlo = np.double(round(xlim[0], 8)).tobytes()   # minimum X value of the grid
            self.xhi = np.double(round(xlim[1], 8)).tobytes()   # maximum X value of the grid
            self.ylo = np.double(round(ylim[0], 8)).tobytes()   # minimum Y value of the grid
            self.yhi = np.double(round(ylim[1], 8)).tobytes()   # maximum Y value of the grid

            self.ncol = np.short(round(n_cells[0], 8))
            self.nrow = np.short(round(n_cells[1], 8))

        elif name == "ncol":    # Esri ASCII raster format -> more information: https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm
            self.ncol = np.intc(f.readline().replace("\n", "").split()[1])  # number of cell columns (int)
            self.nrow = np.intc(f.readline().replace("\n", "").split()[1])  # number of cell rows (int)
            self.xll_name, xll_data = f.readline().replace("\n", "").split()  # X-coordinate of the origin (by center or lower left corner of the cell)
            self.xll_data = np.float32(xll_data)
            self.yll_name, yll_data = f.readline().replace("\n", "").split()  # Y-coordinate of the origin (by center or lower left corner of the cell)
            self.yll_data = np.float32(yll_data)
            self.c_size = np.float32(f.readline().replace("\n", "").split()[1])  # Cell size
            NoData_value = np.float32(f.readline().replace("\n", "").split()[1])
            assert NoData_value < -100 or NoData_value > 10e5, "Change the NoData value (e.g., use -9999)"
            data = f.readlines()
            data = np.asarray([l.replace("\n", "").split() for l in data]).astype(np.float32)

            self.grid_type = 'ESRI_ASCII'
            f.close()
            logger.info("Maps information:\n\t\t\t\t\t\t- Grid type: Esri ASCII\n\t\t\t\t\t\t- Grid size: {}\n\t\t\t\t\t\t- X Min: {}\n\t\t\t\t\t\t- Y Min: {}\n\t\t\t\t\t\t- Number of columns: {}\n\t\t\t\t\t\t- Number of rows: {}\n\t\t\t\t\t\t- Number of cells: {}"
                        .format(self.c_size, round(self.xll_data, 8), round(self.yll_data, 8), self.ncol, self.nrow, data.size))
            return None

        elif name[1] == ".":    # unreferenced ascii file
            data = np.loadtxt(filename, dtype=np.float32)
            f.close()
            self.grid_type = "NREF"
            self.extensions = (0, data.shape[1] - 1, 0, data.shape[0] - 1)
            self.ncol = data.shape[1]
            self.nrow = data.shape[0]
            logger.warning("Reading unreferenced ascii file!")
            return None
        else:
            f.close()
            raise RuntimeError("ERROR in read_map! Filename {} --> wrong file format! \n".format(filename))
        # --------
        f.close()
        self.extensions = (round(xlim[0], 8), round(xlim[1], 8), round(ylim[0], 8), round(ylim[1], 8))

        self.grid_type = name
        self.grid_size = c_size
        assert self.grid_size != 0, "Grid size equal to 0"
        
        logger.info("Maps information:\n\t\t\t\t\t\t- Grid type: {}\n\t\t\t\t\t\t- Grid size: {}\n\t\t\t\t\t\t- X Minimum: {}\n\t\t\t\t\t\t- X Maximum: {}\n\t\t\t\t\t\t- Y Minimum: {}\n\t\t\t\t\t\t- Y Maximum: {}\n\t\t\t\t\t\t- Number of cells: {}"
                    .format(name, c_size, round(xlim[0], 8), round(xlim[1], 8), round(ylim[0], 8), round(ylim[1], 8), data.size))

    def read_map(self, filename):
        ''' 
        Read a file in ASCII or binary format and normalize it.
        Args: 
            filename (string): path to the file to read
        Return:
            data (array): loaded map
        '''

        f = open (filename, 'r', errors="ignore")
        name = f.read(4)
    
        if name == "DSBB":    # file Surfer binary v6
            f.close()
            f = open (filename, 'rb')  # open file in binary mode
            n_cells = np.fromfile(f, dtype=np.short, count=2, offset=4)  # number of cells (X and Y directions)

            f.seek(56, 0)
            # read the matrix with dimensions: n_cells[1] x n_cells[0] of float32
            data = np.reshape(np.fromfile(f, dtype=np.float32), (n_cells[1], n_cells[0]))

        elif name == "DSRB":    # file Surfer binary v7
            f.close()
            f = open (filename, 'rb')  # open file in binary mode
            n_cells = np.fromfile(f, dtype=np.int32, count=2, offset=20)     # number of cells (Y and X directions) => skip 20 bytes (5 long of 4 bytes)

            f.seek(100, 0)
            # read the matrix with dimensions: n_cells[0] x n_cells[1] of double
            data = np.reshape(np.fromfile(f, dtype=np.double), (n_cells[0], n_cells[1])).astype(np.float32)

        elif name == "ncol":    # Esri ASCII raster format
            data = f.readlines()[6:] # skip 6 lines
            data = np.asarray([l.replace("\n", "").split() for l in data]).astype(np.float32)

        elif name == "DSAA":      # Surfer ascii file format
            data = f.readlines()[5:] # skip 5 lines
            data = np.asarray([l.replace("\n", "").split() for l in data]).astype(np.float32)

        elif name[1] == ".":    # unreferenced ascii file
            data = np.loadtxt(filename, dtype=np.float32)
        else:
            f.close()
            raise RuntimeError("ERROR in read_map! Filename {} --> wrong file format! \n".format(filename))
        # --------
        f.close()

        data[data < self.zero_depth] = 0.0
        # --- Outliers  =>  Returns a tensor with NAN = 0
        data = self.filter_nan(data)

        # --- Normalize data
        data = self.map_normalize(data)

        return data
    
    def write_map(self, map, foldername, map_name):
        ''' 
        Write the map in a file.
        Args: 
            map (tensor): array to write in a file.
            foldername (string): path to the folder.
            map_name (string): Name of the map.
        '''
        assert map.shape[0] == self.nrow and map.shape[1] == self.ncol, "Try to save a new Surfer map of size ({}, {}) but the original size is ({}, {})".format(map.shape[0], map.shape[1], self.nrow, self.ncol)
        map = map.cpu().detach().numpy()

        if self.grid_type == "DSBB" or self.grid_type == "DSRB":  # file Surfer binary v6 or V7 -> save as Surfer binary v6
            zlo = np.double(np.amin(map)).tobytes()  # minimum Z value of the grid. NoData nodes are not included in the minimum.
            zhi = np.double(np.amax(map)).tobytes()  # maximum Z value of the grid. NoData nodes are not included in the maximum.

            map = map.tobytes()

            with open(os.path.join(foldername, map_name), "wb") as f:
                f.write(b'DSBB')   # id of the Surfer 6 binary grid
                f.write(self.nx)   # number of grid lines along the X axis (columns)
                f.write(self.ny)   # number of grid lines along the Y axis (rows)
                f.write(self.xlo)  # minimum X value of the grid
                f.write(self.xhi)  # maximum X value of the grid
                f.write(self.ylo)  # minimum Y value of the grid
                f.write(self.yhi)  # maximum Y value of the grid
                f.write(zlo)       # minimum Z value of the grid
                f.write(zhi)       # maximum Z value of the grid
                f.write(map)       # map

        elif self.grid_type == "DSAA":  # Surfer ascii file format
            zlo = np.double(np.amin(map))  # minimum Z value of the grid. NoData nodes are not included in the minimum.
            zhi = np.double(np.amax(map))  # maximum Z value of the grid. NoData nodes are not included in the maximum.

            with open(os.path.join(foldername, map_name), "w") as f:
                f.write("DSAA\n")                              # id of the Surfer ascii grid
                f.write("{} {}\n".format(self.nx, self.ny))    # number of grid lines along the X axis (columns) and Y axis (rows)
                f.write("{} {}\n".format(self.xlo, self.xhi))  # minimum and maximum value of X in the grid
                f.write("{} {}\n".format(self.ylo, self.yhi))  # minimum and maximum value of Y in the grid
                f.write("{} {}\n".format(0, 0))                # minimum and maximum value of Z in the grid
                np.savetxt(f, map)

        elif self.grid_type == "ESRI_ASCII":    # Esri ASCII raster format
            with open(os.path.join(foldername, map_name), "w") as f:
                f.write(f"ncols\t{self.ncol}\n")
                f.write(f"nrows\t{self.nrow}\n")
                f.write(f"{self.xll_name}\t{self.xll_data}\n")
                f.write(f"{self.yll_name}\t{self.yll_data}\n")
                f.write(f"cellsize\t{self.c_size}\n")
                f.write("NODATA_value\t-9999\n")
                np.savetxt(f, map)

        elif self.grid_type == "NREF":
            # Unreferenced ascii file
            np.savetxt(os.path.join(foldername, map_name), map)
        else:
            return

    def write_maps(self, target, preds, save_dir, desc, past=None, iterXbatch=0):
        ''' 
        Help function to write maps in file.
        Args:
            target (tensor): target future frames from the current batch (N, Tf, C, H, W).
            preds (tensor): predicted future frames from the current batch (N, Tf, C, H, W).
            save_dir (path): path to the directory to store the maps.
            desc (string).
            past (tensor): past frames from the current batch (N, Tp, C, H, W).
            iterXbatch (int): iteration x batch size.
        '''
        diff = preds-target  # Map of the differences between predicted and target frames

        for b in range(target.shape[0]):
            foldername = os.path.join(save_dir, "{}_iter_{}".format(desc, b + iterXbatch))
            if not os.path.isdir(foldername):
                os.mkdir(foldername)

            # Write past maps
            if past is not None:
                for i, frame in enumerate(past[b,:,0,:,:]):
                    name = "Past_t{}.grd".format(i + 1 - past.shape[1])
                    self.write_map(frame, foldername, name)

            # Write target maps
            for i, frame in enumerate(target[b,:,0,:,:]):
                name = "True_t{}.grd".format(i + 1)
                self.write_map(frame, foldername, name)

            # Write preds maps
            for i, frame in enumerate(preds[b,:,0,:,:]):
                name = "Pred_t{}.grd".format(i + 1)
                self.write_map(frame, foldername, name)

            # Write difference maps (predicted - target)
            for i, frame in enumerate(diff[b,:,0,:,:]):
                name = "Diff_t{}.grd".format(i + 1)
                self.write_map(frame, foldername, name)

    def write_maps_AE(self, target, preds, save_dir, desc, cur_iter):
        ''' 
        Help function to write maps in file.
        Args: 
            target (tensor): target future frames from the current batch (N, T=1, C, H, W).
            preds (tensor): predicted future frames from the current batch (N, T=1, C, H, W).
            save_dir (path): path to the directory to store the maps.
            desc (string).
            cur_iter (int): current iteration.
        '''
        diff = preds-target  # Map of the differences between predicted and target frames
        foldername = os.path.join(save_dir, "{}_iter_{}".format(desc, cur_iter))
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

        for b in range(target.shape[0]):
            # Write target maps
            name = "Target_batch{}.grd".format(b)
            self.write_map(target[b,0,0,:,:], foldername, name)

            # Write preds maps
            name = "Pred_batch{}.grd".format(b)
            self.write_map(preds[b,0,0,:,:], foldername, name)

            # Write difference maps (predicted - target)
            name = "Diff_batch{}.grd".format(b)
            self.write_map(diff[b,0,0,:,:], foldername, name)

    def filter_nan(self, map):
        ''' Returns an array with NAN = 0 '''
        map[np.isnan(map)] = 0.0
        map[map > 10e5] = 0.0 # Set to 0 the Surfer’s NoData value = 1.71041e38
        return map

    def map_normalize(self, map):
        """
        Normalize a given map between 0 and 1.

        map: map to normalize.
        self.max_depth: maximum water depth of the dataset.
        """
        map = map / self.max_depth
        return map

    def revert_map_normalize(self, map):
        """
        Revert normalization for a given map.

        map: map to revert normalization.
        self.max_depth: maximum water depth of the dataset.
        """
        map = map * self.max_depth
        return map
