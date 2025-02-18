import numpy as np
from scipy.io import loadmat

from src.caching import cache_last_n_files
from src.my_types import ArrayFloat32MxN, ArrayFloat32Nx2
from src.particles import NeighboringParticles


class VelocityDataReader:
    def read_raw(self, file_path: str) -> tuple[ArrayFloat32MxN, ArrayFloat32MxN]:
        """
        Reads velocity data from a MATLAB file and returns it as a tuple of numpy
        arrays (grids of velocity_x and velocity_y) with shapes [M, N].

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            tuple[ArrayFloat32MxN, ArrayFloat32MxN]: Tuple of arrays of shape [M, N].
        """
        data = loadmat(file_path)
        velocity_x = data["velocity_x"]
        velocity_y = data["velocity_y"]
        return velocity_x, velocity_y

    def read_flatten(self, file_path: str) -> ArrayFloat32Nx2:
        """
        Reads velocity data from a MATLAB file and returns it as a numpy array
        with shape [n_points, 2] (velocity_x, velocity_y).

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            ArrayFloat32Nx2: Array of shape [n_points, 2].
        """
        data = loadmat(file_path)
        velocity_x = data["velocity_x"].flatten()
        velocity_y = data["velocity_y"].flatten()
        return np.column_stack((velocity_x, velocity_y))


class CoordinateDataReader:
    def read_raw(self, file_path: str) -> tuple[ArrayFloat32MxN, ArrayFloat32MxN]:
        """
        Reads coordinate data from a MATLAB file and returns it as a tuple of numpy
        arrays (grids of coordinate_x and coordinate_y) with shapes [M, N].

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            tuple[ArrayFloat32MxN, ArrayFloat32MxN]: Tuple of arrays of shape [M, N].
        """
        data = loadmat(file_path)
        coordinate_x = data["coordinate_x"]
        coordinate_y = data["coordinate_y"]
        return coordinate_x, coordinate_y

    def read_flatten(self, file_path: str) -> ArrayFloat32Nx2:
        """
        Reads coordinate data from a MATLAB file and returns it as a numpy array
        with shape [n_points, 2] (coordinate_x, coordinate_y).

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            np.ndarray: Array of shape [n_points, 2].
        """
        data = loadmat(file_path)
        coordinate_x = data["coordinate_x"].flatten()
        coordinate_y = data["coordinate_y"].flatten()
        return np.column_stack((coordinate_x, coordinate_y))


@cache_last_n_files(num_cached_files=2)
def read_seed_particles_coordinates(file_path: str) -> NeighboringParticles:
    """
    Reads seeded particle coordinates from a MATLAB file containing `left`, `right`
    `top` and `bottom` keys to identify the 4 neighboring particles. Then, returns
    a NeighboringParticles object that holds the coordinate array and other
    useful attributes.

    Args:
        file_path (str): Path to the MATLAB file.

    Returns:
        NeighboringParticles: Dataclass of neighboring particles.
    """
    data = loadmat(file_path)
    positions = np.stack(
        [data["left"], data["right"], data["top"], data["bottom"]], axis=1
    )
    positions = positions.reshape(-1, 2, order="F")  # Convert (N, 4, 2) → (4*N, 2)

    return NeighboringParticles(positions=positions)
