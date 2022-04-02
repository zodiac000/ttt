import numpy as np
'''
bbox = [x1, y1, x2, y2]
'''
def compute_iou(bbox1, bbox2):
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    dx = max(xmax - xmin, 0)
    dy = max(ymax - ymin, 0)
    u_area = dx * dy
    area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = u_area / (area - u_area)
    return iou

def is_point_in_triangle(p, p0, p1, p2):
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = p - p0

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inver_deno = 0
    if abs(dot00*dot11 - dot01*dot01) > 1e-6:
        inver_deno = 1/(dot00*dot11 - dot01*dot01)
    u = (dot11*dot02 - dot01*dot12)*inver_deno
    v = (dot00*dot12 - dot01*dot02)*inver_deno

    weight = [1 - u - v, v, u]

    return u >= 0 and v >= 0 and (u+v) < 1, weight

def sparse_to_dense_in_line(pts, dx=1):

    def interpolation(p0, p1, dx):
        v_x = p1[0] - p0[0]
        v_y = p1[1] - p0[1]
        v_l = np.sqrt(v_x*v_x + v_y * v_y)
        if v_l < dx:
            print('xxxxxxxxxxxxxxxxxx')
            return [p0]
        v_x /= v_l
        v_y /= v_l

        if abs(v_x) > abs(v_y):
            s_x = dx / v_x
            s_y = s_x * v_y
            p_num = int((abs((p1[0] - p0[0]) / s_x)))
        else :
            s_y = dx / v_y
            s_x = s_y * v_x
            p_num = int((abs((p1[1] - p0[1]) / s_y)))
        # print(s_x, s_y)
        pts = []
        for i in range(p_num):
            x = p0[0] + i * s_x
            y = p0[1] + i * s_y
            pts.append([x, y])
        return pts

    dense_pts = []
    for i in range(1, len(pts)):
        dense_pts += interpolation(pts[i], pts[i - 1], dx)
    dense_pts.append(pts[-1])
    return dense_pts

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2, enable_backculling):
    epsilon = 1e-6
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(ray_direction, v0v2)
    det = np.dot(v0v1, pvec)
    if enable_backculling:
        if det < epsilon:
            return False, None
    else:
        if abs(det) < epsilon:
            return False, None
    inv_det = 1/det
    tvec = ray_origin - v0

    u = np.dot(tvec, pvec) * inv_det
    if u < 0 or u > 1:
        return False, None
    qvec = np.cross(tvec, v0v1)
    v = np.dot(ray_direction, qvec) * inv_det
    if v < 0 or u+v > 1:
        return False, None
    t = np.dot(v0v2, qvec) * inv_det
    return True, t

def cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    return [theta, r, z]

def is_digit(line):
    return line.lstrip('-').replace('.', '').isdigit()

def line_line_intersect(p0, s0, p1, s1):
    p = p1 - p0
    t1 = (s0[1] * p[0] - s0[0] * p[1]) / (s0[0] * s1[1] - s1[0]*s0[1])
    return p1 + t1 * s1

def point_distance(p0, p1):
    return np.linalg.norm(p0 - p1)

def trajectory_alignment(x, y):
    """
    https://zhuanlan.zhihu.com/p/364825667
    https://github.com/MichaelGrupp/evo/blob/310d49fa107637e11e38cb4bf3ba25b42ec3e15e/evo/core/trajectory.py#L185
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points 3Xn
    :param y: mxn matrix of points, m = dimension, n = nr. of data points 3Xn
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    y = c * r * x + t
    """
    if x.shape != y.shape:
        raise ValueError("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("Degenerate covariance rank, "
                   "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def nearest_distance2d(pts):
    x = pts[0][0]-pts[1][0]
    y = pts[0][1]-pts[1][1]
    return np.sqrt(x*x+y*y)

def nearest_point2d_pair_brute_force(s, dis_thres=1, distance_func=nearest_distance2d):
    s_len = len(s)
    if s_len < 2:
        return []

    point_pairs = []
    nearest_point_pair = [0, 1, s[0], s[1]]
    min_dis = distance_func(nearest_point_pair[2:])
    for i in range(s_len):
        for j in range(i+1, s_len):
            dis = distance_func([s[i], s[j]])
            if dis < dis_thres:
                point_pairs.append([i, j, s[i], s[j]])
            if dis < min_dis:
                nearest_point_pair = [[i, j, s[i], s[j]]]
                min_dis = dis
    return nearest_point_pair, point_pairs

def nearest_point2d_pair_divide(s, distance_func=nearest_distance2d):
    mid = int(len(s) / 2)
    left = s[0:mid]
    right = s[mid:]
    mid_x = (left[-1][0] + right[0][0]) / 2.0
    if len(left) > 2:
        lmin = nearest_point2d_pair_divide(left)  # 左侧部分最近点对
    else:
        lmin = left
    if len(right) > 2:
        rmin = nearest_point2d_pair_divide(right)  # 右侧部分最近点对
    else:
        rmin = right
    if len(lmin) > 1:
        dis_l = distance_func(lmin)
    else:
        dis_l = float("inf")
    if len(rmin) > 1:
        dis_r = distance_func(rmin)
    else:
        dis_r = float("inf")
    d = min(dis_l, dis_r)  # 最近点对距离
    mid_min = []
    for i in left:
        if mid_x - i[0] <= d:  # 如果左侧部分与中间线的距离<=d
            for j in right:
                if abs(i[0] - j[0]) <= d and abs(i[1] - j[1]) <= d:  # 如果右侧部分点在i点的(d,2d)之间
                    if distance_func((i, j)) <= d: mid_min.append([i, j])  # ij两点的间距若小于d则加入队列
    if mid_min:
        dic = []
        for i in mid_min:
            dic.append({distance_func(i): i})
        dic.sort(key=lambda x: x.keys())
        return list(dic[0].values())[0]
    elif dis_l > dis_r:
        return rmin
    else:
        return lmin

