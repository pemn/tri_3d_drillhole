#!python
# create a 3d triangulation of a drillhole
# optionally with photometry texture
# input_header: csv with hole collar
# input_survey: csv with hole survey
# input_texture: image with downhole photometry
# output_path: path to save result as Wavefront obj
# open_spiral: create a open spiral instead of cylinder (cinnamon bark shape)
# closed_tube: create a closed tube by adding covers at start and end
# display: show result in a 3d window
# v1.0 06/2021 paulo.ernesto
# -
# Copyright 2021 Vale

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

'''
usage: $0 input_header*xlsx,csv input_survey*csv,xlsx input_texture*xlsx,jpg,png output*obj,glb open_spiral@ closed_tube@ display@
'''
import sys
import pandas as pd
import numpy as np

import scipy.linalg

import skimage.io
import struct
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, pd_load_dataframe, pd_save_dataframe, pd_synonyms, nodes_faces_to_df
from pd_vtk import vtk_nf_to_mesh, pv_save, vtk_to_gltf
from surveyholes import desurvey_hole
import pyvista as pv
from ml_append_trough import wb_append_trough

def matrix_rotate_3d(axis, theta):
  '''
  let a be the unit vector along axis, i.e. a = axis/norm(axis)
  and A = I × a be the skew-symmetric matrix associated to a, i.e. the cross product of the identity matrix with a

  then M = exp(θ A) is the rotation matrix.
  '''
  return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta))

def texture_map_to_cylinder(n_points, width_resolution):
  n_cilinders = int(n_points / width_resolution)
  t_coord_x = np.tile(np.linspace(0, 1, width_resolution), n_cilinders)
  t_coord_y = np.repeat(np.linspace(1, 0, n_cilinders), width_resolution)
  return np.stack([t_coord_x, t_coord_y], 1)

def triangulate_3d_cylinder(rows, cols):
  ' weave faces for a cylinder point cloud '
  faces = []

  for i in range(1, rows):
    ni = i - 1
    for j in range(cols):
      nj = j + 1
      if nj == cols:
        nj = 0
      faces.append([i * cols + j , i  * cols + nj , ni * cols + j])
      faces.append([i * cols + nj,  ni * cols + j, ni * cols + nj])

  return faces

def create_3d_hole(hdf, sdf, survey, width_resolution = 36, radius = 1., open_spiral = False, closed_tube = False, xyz = ['x','y','z']):

  # create linspace using rounded intervals
  theta = np.linspace(0, 2*np.pi, width_resolution, False)

  theta_grid, z_grid = np.meshgrid(theta, np.arange(survey.shape[0], dtype=np.float_))
  
  x_grid = np.zeros(theta_grid.shape)
  y_grid = np.zeros(theta_grid.shape)

  #t_coord_x = np.tile(np.linspace(0, 1, width_resolution), survey.shape[0])
  #t_coord_y = np.repeat(np.linspace(1, 0, survey.shape[0]), width_resolution)
  t_coord_y = np.ndarray(survey.shape[0], dtype=np.float32)

  t = 0
  for k in range(survey.shape[0]):
    if k:
      dv = survey[k, 1:4] - survey[k-1, 1:4]
    elif survey.shape[0] > 1:
      dv = survey[k+1, 1:4] - survey[k, 1:4]
    else:
      dv = [0, 0, -1]
    # accumulate the total length
    t += np.linalg.norm(dv)
    t_coord_y[k] = t

    for j in range(theta.size):
      # create a rotation matrix relative to the segment normal
      mr = matrix_rotate_3d(dv, theta[j])
      hand_size = radius
      if open_spiral:
        hand_size *= (width_resolution - 1) / width_resolution + (1/width_resolution * j / theta.size)

      # apply the rotation matrix to the "clock hand" vector
      v_delta = np.dot(mr, [hand_size, 0., 0.])

      x_grid[k, j] = survey[k, 1] + v_delta[0]
      y_grid[k, j] = survey[k, 2] + v_delta[1]
      z_grid[k, j] = survey[k, 3] + v_delta[2]

  # add the real word coordinate of hole collar to the relative vector
  x_grid += hdf[xyz[0]]
  y_grid += hdf[xyz[1]]
  z_grid += hdf[xyz[2]]

  xyz_nodes = np.column_stack((x_grid.flat, y_grid.flat, z_grid.flat))

  # normalize from T to 1-0
  t_coord_y /= t
  # np.abs(np.linspace(-1, 1, 9, endpoint=True))
  # x goes around horizontal in the hole
  #t_coord_x = np.tile(np.linspace(0, 1, survey.shape[0]), width_resolution)
  #t_coord_x = np.tile(np.linspace(0, 1, width_resolution), survey.shape[0])
  t_coord_x = np.tile(np.abs(np.linspace(-1, 1, width_resolution, dtype=np.float32)), survey.shape[0])
  # y goes downhole
  t_coord_y = np.repeat(t_coord_y, width_resolution)
  #t_coord_y = np.tile(t_coord_y, width_resolution)

  faces = triangulate_3d_cylinder(survey.shape[0], width_resolution)
  #print(faces)
  if closed_tube:
    #print(xyz_nodes[0:width_resolution])
    tri = Delaunay([xyz_nodes[i][:2] for i in range(width_resolution)])
    #faces.extend(tri.simplices)
    #print()
    faces = list(tri.simplices) + faces + np.add(list(tri.simplices), width_resolution * (survey.shape[0] - 1)).tolist()
    #print(Delaunay(xyz_nodes[-width_resolution:]))

  return xyz_nodes, faces, np.stack([t_coord_x, t_coord_y], 1)

def plot_3d_drillhole():
  fig = plt.figure()

  ax = plt.subplot(121, projection='3d', azim=0, elev=30)
  ax.plot(survey[:, 1], survey[:, 2], survey[:, 3])
  # ax = Axes3D(fig, azim=0, elev=60)
  ax = plt.subplot(122, projection='3d', azim=0, elev=30)

  ax.plot_trisurf(nodes[:,0], nodes[:,1], nodes[:,2], triangles=faces, cmap=plt.cm.Spectral)
  old_lim = [ax.get_xlim(), ax.get_ylim()]
  ax.set_xlim(np.min(old_lim, 0)[0], np.max(old_lim, 0)[1])
  ax.set_ylim(ax.get_xlim())

  plt.show()

def img_to_texture(img, hid = None):
  out = img
  if isinstance(img, dict):
    if hid and hid in img:
      out = img[hid]
  
  out = np.flip(out, 0)

  return pv.Texture(out)

def tri_3d_drillhole(input_header, input_survey, input_texture, output, open_spiral, closed_tube, display):
  ''' 
  doc
  '''
  # hardcoded constants
  width_resolution = 32
  cylinder_radius = .1

  hdf = pd_load_dataframe(input_header)
  sdf = pd_load_dataframe(input_survey)

  v_hid = pd_synonyms(sdf, ["furo","hid","dhid","bhid"])
  v_depth = pd_synonyms(sdf, ["depth", "prof","at"])
  v_azimuth = pd_synonyms(sdf, ["azimuth", "azim", "azi","brg"])
  v_dip = pd_synonyms(sdf, ["dip", "inclin"])
  v_x = pd_synonyms(hdf, ["x", "east", "leste"])
  v_y = pd_synonyms(hdf, ["y", "north", "norte"])
  v_z = pd_synonyms(hdf, ["z", "level", "cota"])

  hdf.set_index(v_hid, inplace=True)
  sdf.set_index(v_hid, inplace=True)
  odf = pd.DataFrame()
  meshes = []
  img = None
  if input_texture:
    if input_texture.lower().endswith('xlsx'):
      img = wb_append_trough(input_texture, True)
    else:
      img = skimage.io.imread(input_texture)

  for hid in hdf.index.unique():
    print(hid)
    sdf_hid = sdf.loc[hid]
    if np.ndim(sdf_hid) == 1:
      sdf_hid = pd.DataFrame.from_records([sdf_hid])
    survey = desurvey_hole(sdf_hid[v_depth], sdf_hid[v_azimuth], sdf_hid[v_dip] * -1, False)

    nodes, faces, coord = create_3d_hole(hdf.loc[hid], sdf_hid, survey, width_resolution, cylinder_radius, int(open_spiral), int(closed_tube), [v_x, v_y, v_z])

    df = nodes_faces_to_df(nodes, faces)
    df['hid'] = hid
    odf = odf.append(df)
    mesh = vtk_nf_to_mesh(nodes, faces)
    if img is not None:
      mesh.t_coords = coord
      mesh.textures[0] = img_to_texture(img, hid)

    meshes.append(mesh)

  if output:
    if output.lower().endswith('vtk'):
      pv_save(meshes, output, False)
    elif output.lower().endswith('glb'):
      gltf = vtk_to_gltf(meshes)
      gltf.save(output)
    else:
      pd_save_dataframe(odf, output)
  else:
    print(odf.to_string())

  if int(display):
    p = pv.Plotter()
    p.add_axes()
    for mesh in meshes:
      p.add_mesh(mesh, texture=True)
      #p.add_point_labels(mesh.points, np.arange(mesh.n_points))
    p.show()

  print("finished")

main = tri_3d_drillhole

if __name__=="__main__":
  usage_gui(__doc__)

# python tri_3d_drillhole.py C:/home/code/python/holebook/FD00120_header.csv C:/home/code/python/holebook/FD00120_survey.csv "" "" 0 1
