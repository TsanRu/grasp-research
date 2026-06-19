import numpy as np
import quaternion


def orientation_error_as_rotation_vector(quat_target, quat_source):
    """
    Compute the orientation error between two quaternions as a rotation vector.

    Args:
        quat_target (np.ndarray): The target quaternion.
        quat_source (np.ndarray): The source quaternion.

    Returns:
        np.ndarray: The rotation vector representing the orientation error.
    """
    qt = to_np_quaternion(quat_target)
    qs = to_np_quaternion(quat_source)
    return quaternion.as_rotation_vector(qt*qs.conjugate())


def quaternions_orientation_error(quat_target, quat_source):
    """
    Compute the orientation error between two quaternions.

    Args:
        quat_target (np.ndarray): The target quaternion.
        quat_source (np.ndarray): The source quaternion.

    Returns:
        np.ndarray: The quaternion representing the orientation error.
    """
    qt = to_np_quaternion(quat_target)
    qs = to_np_quaternion(quat_source)
    return to_np_array(qt*qs.conjugate())

# Quaternion Math


def quaternion_normalize(q):
    """
    Normalize a quaternion.

    Args:
        q (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The normalized quaternion.
    """
    np_q = to_np_quaternion(q)
    return to_np_array(np_q.normalized())


def quaternion_conjugate(quaternion):
    """
    Return the conjugate of a quaternion.

    Args:
        quaternion (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The conjugate of the input quaternion.
    """
    return np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)


def quaternion_inverse(quaternion):
    """
    Return the inverse of a quaternion.

    Args:
        quaternion (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The inverse of the input quaternion.
    """
    return to_np_array(to_np_quaternion(quaternion).inverse())


def quaternion_multiply(quaternion1, quaternion0):
    """
    Multiply two quaternions.

    Args:
        quaternion1 (np.ndarray): The first quaternion.
        quaternion0 (np.ndarray): The second quaternion.

    Returns:
        np.ndarray: The result of multiplying the two quaternions.
    """
    q1 = to_np_quaternion(quaternion1)
    q0 = to_np_quaternion(quaternion0)
    return to_np_array(q1*q0)


def quaternion_slerp(quat0, quat1, fraction):  # TODO
    """
    Perform spherical linear interpolation (SLERP) between two quaternions.

    Args:
        quat0 (np.ndarray): The first quaternion.
        quat1 (np.ndarray): The second quaternion.
        fraction (float): The interpolation fraction (between 0 and 1).

    Returns:
        np.ndarray: The interpolated quaternion.
    """
    q0 = to_np_quaternion(quat0)
    q1 = to_np_quaternion(quat1)
    return to_np_array(quaternion.slerp(q0, q1, 0.0, 1.0, fraction))


def quaternion_rotate_vector(quat, vector):
    """
    Rotate a vector using a given unit quaternion.

    Args:
        quat (np.ndarray): The input quaternion.
        vector (np.ndarray): The vector to be rotated.

    Returns:
        np.ndarray: The rotated vector.
    """
    q = to_np_quaternion(quat)
    return quaternion.rotate_vectors(q, vector)


def diff_quaternion(quat1, quat2):
    """
    Compute the difference between two quaternions.

    Args:
        quat1 (np.ndarray): The first quaternion.
        quat2 (np.ndarray): The second quaternion.

    Returns:
        np.ndarray: The quaternion representing the difference between the two input quaternions.
    """
    q1 = to_np_quaternion(quat1)
    q2 = to_np_quaternion(quat2)
    return to_np_array(q2*q1.inverse())


def random_quaternion(rand=None):
    """
    Generate a random quaternion.

    Args:
        rand (np.ndarray, optional): An array of 3 random numbers between 0 and 1. If not provided, random numbers will be generated.

    Returns:
        np.ndarray: A random quaternion.
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1)*r1, np.cos(t1)*r1, np.sin(t2)*r2, np.cos(t2)*r2), dtype=np.float64)


def random_rotation_matrix(rand=None):
    """
    Generate a random rotation matrix.

    Args:
        rand (np.ndarray, optional): An array of 3 random numbers between 0 and 1. If not provided, random numbers will be generated.

    Returns:
        np.ndarray: A random rotation matrix.
    """
    return rotation_matrix_from_quaternion(random_quaternion(rand))

# Conversion


def rotation_matrix_from_quaternion(q):
    """
    Convert a quaternion to a 4x4 rotation matrix.

    Args:
        q (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The 4x4 rotation matrix.
    """
    R = np.eye(4)
    R[:3, :3] = quaternion.as_rotation_matrix(to_np_quaternion(q))
    return R


def quaternion_from_matrix(matrix):
    """
    Converts a 3x3 rotation matrix to a 4-element quaternion representation.

    Args:
        matrix (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        numpy.ndarray: A 4-element numpy array representing the quaternion.
    """
    q = quaternion.from_rotation_matrix(matrix[:3, :3])
    return to_np_array(q)


def quaternion_from_axis_angle(axis_angle):
    """
    Convert an axis-angle representation to a quaternion.

    Args:
        axis_angle (np.ndarray): The axis-angle representation.

    Returns:
        np.ndarray: The corresponding quaternion.
    """
    np_q = quaternion.from_rotation_vector(axis_angle)
    return to_np_array(np_q)


def quaternion_from_ortho6(ortho6):
    """
    Convert an orthographic 6D representation to a quaternion.

    Args:
        ortho6 (np.ndarray): The orthographic 6D representation.

    Returns:
        np.ndarray: The corresponding quaternion.
    """
    R = rotation_matrix_from_ortho6(ortho6)
    return quaternion_from_matrix(R)


def axis_angle_from_quaternion(quat):
    """
    Convert a quaternion to an axis-angle representation.

    Args:
        quat (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The corresponding axis-angle representation.
    """
    return quaternion.as_rotation_vector(to_np_quaternion(quat))


def ortho6_from_axis_angle(axis_angle):
    """
    Convert an axis-angle representation to an orthographic 6D representation.

    Args:
        axis_angle (np.ndarray): The axis-angle representation.

    Returns:
        np.ndarray: The corresponding orthographic 6D representation.
    """
    return ortho6_from_quaternion(quaternion_from_axis_angle(axis_angle))


def ortho6_from_quaternion(q):
    """
    Convert a quaternion to an orthographic 6D representation.

    Args:
        q (np.ndarray): The input quaternion.

    Returns:
        np.ndarray: The corresponding orthographic 6D representation.
    """
    R = rotation_matrix_from_quaternion(q)
    return R[:3, :2].T.flatten()


def rotation_matrix_from_ortho6(ortho6):
    """
    Convert an orthographic 6D representation to a rotation matrix.

    Args:
        ortho6 (np.ndarray): The orthographic 6D representation.

    Returns:
        np.ndarray: The corresponding rotation matrix.
    """
    x_raw, y_raw = ortho6[0:3], ortho6[3:6]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.eye(4)
    R[:3, :3] = np.column_stack((x, y, z))
    return R


def to_np_quaternion(q: np.ndarray) -> np.quaternion:
    """
    Convert a numpy array to a numpy quaternion.

    Args:
        q (np.ndarray): The input numpy array.

    Returns:
        np.quaternion: The corresponding numpy quaternion.
    """
    return np.quaternion(q[3], q[0], q[1], q[2])


def to_np_array(np_q: np.quaternion) -> np.ndarray:
    """
    Convert a numpy quaternion to a numpy array.

    Args:
        np_q (np.quaternion): The input numpy quaternion.

    Returns:
        np.ndarray: The corresponding numpy array.
    """
    return np.array([np_q.x, np_q.y, np_q.z, np_q.w])
