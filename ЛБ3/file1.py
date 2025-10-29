import numpy as np
from numpy import sin, cos
from PIL import Image, ImageOps

img_mat=np.zeros((1000,1000,3),dtype=np.uint8)
img_mat_z_buf=np.full((1000,1000), np.inf)

def bar_coord(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_trian(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat, img_mat_z_buf, color):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > 999): xmax = 999
    if (ymax > 999): ymax = 999
    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            lambda0, lambda1, lambda2 = bar_coord(x, y, x0, y0, x1, y1, x2, y2)
            if ((lambda0 > 0) and (lambda1 > 0) and (lambda2 > 0)):
                z = z0 * lambda0 + z1 * lambda1 + z2 * lambda2
                if (z < img_mat_z_buf[x, y]):
                    img_mat_z_buf[x, y] = z
                    img_mat[y, x] = color

def light_incidence_angle_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2) :
    v0 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v1 = np.array([x1 - x0, y1 - y0, z1 - z0])
    n = np.cross(v0, v1)
    l = np.array([0, 0, 1])
    angle_cos = (np.dot(n, l, out = None)) / (np.linalg.norm(l) * np.linalg.norm(n))
    return angle_cos

def rotation_and_shift(start_coords, x_angle, y_angle, z_angle, tx, ty, tz):
    
    shift_vector = np.array([tx, ty, tz])
    rotation_matrix_x = np.array([[1, 0, 0], [0, cos(x_angle), sin(x_angle)], [0, -sin(x_angle), cos(x_angle)]])
    rotation_matrix_y = np.array([[cos(y_angle), 0, sin(y_angle)], [0, 1, 0], [-sin(y_angle), 0, cos(y_angle)]])
    rotation_matrix_z = np.array([[cos(z_angle), sin(z_angle), 0], [-sin(z_angle), cos(z_angle), 0], [0, 0, 1]])
    rotation_matrix = np.matmul(rotation_matrix_x, np.matmul(rotation_matrix_y, rotation_matrix_z))
    new_coords = np.matmul(rotation_matrix, start_coords) + shift_vector
    return new_coords

def unpack(new_coords):
    return new_coords[0], new_coords[1], new_coords[2]

v = []
f = []
for i in open('E:\обуч\лабораторные\комп граф\-\ЛБ3\model_1.obj'):
    if i[0:2] == 'v ':
        v.append(list(map(float, i.split()[1:])))
    elif i[0:2] == 'f ':
        s = i.split()[1:]
        a = []
        for j in range(3):
            a.append(int(s[j].split('/')[0]))
        f.append(a)


for k in range(len(f)):
    x0 = v[f[k][0]-1][0]
    y0 = v[f[k][0]-1][1]
    z0 = v[f[k][0]-1][2]
    x1 = v[f[k][1]-1][0]
    y1 = v[f[k][1]-1][1]
    z1 = v[f[k][1]-1][2]
    x2 = v[f[k][2]-1][0]
    y2 = v[f[k][2]-1][1]
    z2 = v[f[k][2]-1][2]

    x0, y0, z0 = unpack(rotation_and_shift([x0, y0, z0], 0, 4, 0, 0, -0.05, 0.15))
    x1, y1, z1 = unpack(rotation_and_shift([x1, y1, z1], 0, 4, 0, 0, -0.05, 0.15))
    x2, y2, z2 = unpack(rotation_and_shift([x2, y2, z2], 0, 4, 0, 0, -0.05, 0.15))

    x0_s = 800 * x0 / z0 + 500
    y0_s = 800 * y0 / z0 + 500
    x1_s = 800 * x1 / z1 + 500
    y1_s = 800 * y1 / z1 + 500
    x2_s = 800 * x2 / z2 + 500
    y2_s = 800 * y2 / z2 + 500

    arccosin = light_incidence_angle_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    
    color = [-255 * arccosin, 0, 0]
    if (arccosin < 0): draw_trian(x0_s, y0_s, z0, x1_s, y1_s, z1, x2_s, y2_s, z2, img_mat, img_mat_z_buf, color)

img=Image.fromarray(img_mat,mode='RGB')
img = ImageOps.flip(img)
img.save('img_triangls.png')
