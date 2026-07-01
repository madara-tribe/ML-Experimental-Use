import math
def calculateClampedAngle(x):
    z= 90
    #if 0<= x <=80:
    #    return 0
    angle_rad = math.atan(x / z)
    deg = angle_rad * (180 / math.pi)
    return deg


def calculate_angle_APh(x, z, x_max=80):
    # 座標
    A = (0, 0)
    P = (int(x_max/2), z)
    h = (x, 0)

    # ベクトル AP, hP（Pを共通の始点とする）
    vec_AP = (A[0] - P[0], A[1] - P[1])
    vec_hP = (h[0] - P[0], h[1] - P[1])

    # 内積と大きさ
    dot = vec_AP[0]*vec_hP[0] + vec_AP[1]*vec_hP[1]
    mag_AP = math.hypot(*vec_AP)
    mag_hP = math.hypot(*vec_hP)

    # cosθ = dot / (|a||b|)
    cos_theta = dot / (mag_AP * mag_hP)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


x_list = [0, 10, 20, 30, 40, 50, 60, 70, 80]
for x in x_list:
    print(calculateClampedAngle(x))
print(calculate_angle_APh(23.11, 97.29, 100))
