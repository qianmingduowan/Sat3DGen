import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

def GetPerspective(image, intrinsic_four, yaw, pitch, height, width):
    #
    # for yaw, higher is looking right, lower is looking left
    # for pitch, higher is looking down, lower is looking up
    #
    if not type(image) == list:
        image = [image]
    THETA = yaw
    PHI = -pitch
    fx,fy,cx,cy = intrinsic_four
    K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1],
        ], np.float32)
    K_inv = np.linalg.inv(K)
    
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz) 
    XY = lonlat2XY(lonlat, shape=image[0].shape).astype(np.float32)
    result = []
    for img in image:
        persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        persp = np.clip(persp, 0, 255).astype(np.uint8)
        result.append(persp)
    if len(result) == 1:
        return result[0]
    return result
