import numpy as np
import h5py
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from alpha_shape import alpha_shape_border
from perturb_proj_tree import PerturbProjTree

np.random.seed(1234)

view_label = "chair"
offset_idx = 0
num_points = 1024
f = h5py.File("../data/point_clouds.hdf5", "r")
shape_names = [line.rstrip() for line in open("../data/shape_names.txt")]

pc = f["points"][:][:, :num_points]
labels = f["labels"][:]

f.close()

print("Shape:", pc.shape)
print("Number of points:", num_points)
print("Labels:", [shape_names[idx] for idx in np.unique(labels)])
print("Selected label:", view_label)

match_idx = np.where(labels == shape_names.index(view_label))[0]
view_pc = pc[match_idx[offset_idx]]

print("Shape index:", match_idx[offset_idx])

plt.figure(figsize = (30, 15))

def scale_plot():
    plt.gca().auto_scale_xyz((-1, 1), (-1, 1), (-1, 1))
    plt.gca().view_init(0, 0)
    plt.axis("off")

plt.subplot(121, projection = "3d")

alpha_points, alpha_triangles = alpha_shape_border(view_pc)
alpha_points = alpha_points[:, (0, 2, 1)]

print("Number of points in alpha shape:", alpha_points.shape[0])
print("Number of triangles in alpha shape:", alpha_triangles.shape[0])

plt.gca().plot_trisurf(*alpha_points.T, triangles = alpha_triangles)

rand_perturb = (np.random.random(view_pc.shape) * 0.03 + 0.03) * np.random.choice(np.array([-1, 1]), size = view_pc.shape)
perturbed = view_pc + rand_perturb

colors = np.random.random(num_points)
plt.gca().scatter(*perturbed.T, zdir = "y", s = 300, c = colors, cmap = "rainbow")

scale_plot()

plt.subplot(122, projection = "3d")

plt.gca().plot_trisurf(*alpha_points.T, triangles = alpha_triangles)

print("Projection started.")
start_time = time.time()
tree = PerturbProjTree(view_pc, thickness = 0.01)
projected = tree.project(perturbed, rand_perturb)
end_time = time.time()
print("Projection ended in %f seconds." % (end_time - start_time))

plt.gca().scatter(*projected.T, zdir = "y", s = 300, c = colors, cmap = "rainbow")

scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.show()
