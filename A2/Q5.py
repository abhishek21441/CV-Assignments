import numpy as np
import open3d as o3d
import os

def compute_plane_parameters(points,mean):
    u, s, vh = np.linalg.svd(points)

    normal = vh[-1]
    offset = np.dot(normal, mean)
    if offset <= 0:
        offset *= -1
        normal *= -1
    
    return normal, offset

def compute_all_plane_parameters(pcd_dir):
    lidar_plane_normals = []
    lidar_offsets = []
    for file in os.listdir(pcd_dir):
        if file.endswith(".pcd"):
            pcd_file = os.path.join(pcd_dir, file)
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            mean = np.mean(points, axis = 0)
            points -=  mean
            normal, offset = compute_plane_parameters(points,mean)
            lidar_plane_normals.append(normal)
            lidar_offsets.append(offset)
    lidar_plane_normals = np.array(lidar_plane_normals)
    lidar_offsets = np.array(lidar_offsets)

    return lidar_plane_normals, lidar_offsets

pcd_dir = 'CV-A2-Q5\lidar_scans'

lidar_plane_normals, lidar_offsets = compute_all_plane_parameters(pcd_dir)


print(lidar_plane_normals.shape)
print(lidar_offsets.shape)
print(lidar_offsets)
print(max(lidar_offsets))


def read_camera_normals(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        normal = np.array([float(line.strip()) for line in lines])
    return normal

def read_rotation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        rotation_matrix = np.array([[float(value) for value in line.strip().split()] for line in lines])
    return rotation_matrix

def read_translation_vector(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        translation_vector = np.array([float(line.strip()) for line in lines])
    return translation_vector

camera_dir = 'CV-A2-Q5\camera_parameters'

camera_normals = []
rotation_matrices = []
translation_vectors = []

for frame_folder in os.listdir(camera_dir):
    # print(frame_folder)
    frame_folder_path = os.path.join(camera_dir, frame_folder)
    if os.path.isdir(frame_folder_path):
        # print(frame_folder_path)
        normals_file_path = os.path.join(frame_folder_path, 'camera_normals.txt')
        rotation_matrix_file_path = os.path.join(frame_folder_path, 'rotation_matrix.txt')
        translation_vector_file_path = os.path.join(frame_folder_path, 'translation_vectors.txt')
 
        normal = read_camera_normals(normals_file_path)
        rotation_matrix = read_rotation_matrix(rotation_matrix_file_path)
        translation_vector = read_translation_vector(translation_vector_file_path)

        camera_normals.append(normal)
        rotation_matrices.append(rotation_matrix)
        translation_vectors.append(translation_vector)

camera_normals = np.array(camera_normals)
rotation_matrices = np.array(rotation_matrices)
translation_vectors = np.array(translation_vectors)

print(camera_normals.shape)
print(rotation_matrices.shape)
print(translation_vectors.shape)


new_origins = []

for i in range(len(rotation_matrices)):
    translated_origin = np.array([0, 0, 0]) - translation_vectors[i]

    rotated_origin = np.dot(rotation_matrices[i], translated_origin)

    new_origins.append(rotated_origin)

new_origins = np.array(new_origins)

print(new_origins.shape)


import copy
camera_offsets = []
new_camera_normals = []

for i in range(len(new_origins)):
    camera_normal = camera_normals[i]

    new_origin = new_origins[i]

    offset = -np.dot(camera_normal, new_origin)
    if offset <= 0 :
        offset *= -1
        camera_normal *= -1
        new_camera_normals.append(camera_normal)
    else:
        new_camera_normals.append(camera_normal)
    
    camera_offsets.append(offset)

camera_offsets = np.array(camera_offsets)

camera_normals = np.array(copy.deepcopy(new_camera_normals))
print(camera_normals.shape)
print(camera_offsets.shape)


camera_normals_real = camera_normals.T
print(camera_normals_real.shape)

theta_c_theta_c_transpose = np.matmul(camera_normals_real, camera_normals_real.T)
print(theta_c_theta_c_transpose.shape)

theta_c_theta_c_transpose_inv = np.linalg.inv(theta_c_theta_c_transpose) # -------- (i)
print(theta_c_theta_c_transpose_inv.shape)

offset_difference = camera_offsets - lidar_offsets # ---------- (i)
print(offset_difference.shape)

lidar_translate_vectors = np.matmul(np.matmul(theta_c_theta_c_transpose_inv,camera_normals_real,),offset_difference)

print(lidar_translate_vectors)


lidar_plane_normals_real = lidar_plane_normals.T
u, s, vh = np.linalg.svd(np.matmul(lidar_plane_normals_real,camera_normals)) # ------- (i)
print(np.matmul(lidar_plane_normals_real,camera_normals).shape)
lidar_rotation_matrix = np.matmul(vh.T,u.T)
# lidar_rotation_matrix[-1] *= -1
print(lidar_rotation_matrix.shape)
print(lidar_rotation_matrix)
determinant = np.linalg.det(lidar_rotation_matrix)
print(determinant)


from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

cosine_distances = []
c_norm_est_list = []

for i in range(38):
    l_norm = lidar_plane_normals[i]
    l_rotmat = lidar_rotation_matrix
    c_norm_est = np.matmul(l_rotmat, l_norm)
    c_norm_est_list.append(c_norm_est)
    v1 = c_norm_est.reshape(1, -1)
    v2 = camera_normals[i].reshape(1, -1)

    cosine_sim = cosine_similarity(v1, v2)
    cosine_dist = 1 - cosine_sim

    cosine_distances.append(cosine_dist[0][0])

plt.figure(figsize=(10, 6))
plt.plot(range(1, 39), cosine_distances, marker='o', linestyle='-')
plt.title('Cosine Distances between LIDAR and Camera Normals')
plt.xlabel('Image Number')
plt.ylabel('Cosine Distance')
plt.grid(True)
plt.xticks(range(1, 39))
plt.show()   

average_error = np.mean(cosine_distances)
std_deviation = np.std(cosine_distances)

print("Average Error:", average_error)
print("Standard Deviation:", std_deviation)


import cv2

path = "CV-A2-Q5\camera_parameters\camera_intrinsic.txt"
path2 = "CV-A2-Q5\camera_parameters\distortion.txt"
intrinsic_params = np.loadtxt(path)
dist = np.loadtxt(path2)
intrinsic_params = intrinsic_params.reshape(3, 3)
dist = dist.reshape(5,)

def lidar_points_cam(lidar_points, lidar_rotation_matrix, lidar_translate_vectors, intrinsic_params):
    projected_points = cv2.projectPoints(lidar_points, lidar_rotation_matrix, lidar_translate_vectors, intrinsic_params, dist)[0][:, 0, :]
    return projected_points


pcd_dir = 'CV-A2-Q5/lidar_scans'

lidar_points_list = []

for file in os.listdir(pcd_dir):
    if file.endswith(".pcd"):
        pcd_file = os.path.join(pcd_dir, file)
        # Read the .pcd file
        pcd = o3d.io.read_point_cloud(pcd_file)
        # Convert to numpy array
        points = np.asarray(pcd.points)

        projected_points = lidar_points_cam(points, lidar_rotation_matrix, lidar_translate_vectors, intrinsic_params)
        lidar_points_list.append(projected_points)

print(projected_points.shape)

start_x,start_y = np.mean(projected_points,axis = 0)
print(start_x,start_y)


img_list = ["frame_1061.jpeg", "frame_1075.jpeg", "frame_1093.jpeg", "frame_1139.jpeg", "frame_1153.jpeg"]
img_path = "CV-A2-Q5/camera_images/"
for i in range(5):
    image = cv2.imread(os.path.join(img_path, img_list[i]))
    start_x,start_y = np.mean(lidar_points_list[i],axis = 0)

    end_lidar_point_x = start_x - (lidar_plane_normals[i][0]*100)
    end_lidar_point_y = start_y - (lidar_plane_normals[i][1]*100)

    end_camera_point_x = start_x - (camera_normals[i][0]*100)
    end_camera_point_y = start_y - (camera_normals[i][1]*100)

    end_est_point_x = start_x - (c_norm_est_list[i][0]*100)
    end_est_point_y = start_y - (c_norm_est_list[i][1]*100)
    
    #LIDAR
    cv2.arrowedLine(image,(int(start_x), int(start_y)), (int(end_lidar_point_x),int(end_lidar_point_y)), (255,0,0), 3)

    #CAMERA
    cv2.arrowedLine(image,(int(start_x), int(start_y)), (int(end_camera_point_x),int(end_camera_point_y)), (0,255,0), 3)

    #CAM_ESTIMATE
    cv2.arrowedLine(image,(int(start_x), int(start_y)), (int(end_est_point_x),int(end_est_point_y)), (0,0,255), 3)

    cv2.imwrite(f"Normals{i}.jpeg",image)


img_path = "CV-A2-Q5/camera_images/"

for i in range(5):
    image = cv2.imread(os.path.join(img_path, img_list[i]))
    for j in range(len(lidar_points_list[i])):
        cv2.circle(image, (int(lidar_points_list[i][j][0]), int(lidar_points_list[i][j][1])), 2, (0, 0, 255), -1)
    
    cv2.imwrite(f"Points{i}.jpeg",image)
    