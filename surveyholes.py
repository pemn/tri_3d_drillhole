#!python

import numpy as np
import pandas as pd

def desurvey_hole(depth, phi, theta, matrix = False, downhole = False):
  # get lengths of the separate segments 
  lengths = np.array(depth)
  #np.subtract(depth[1:], depth[:-1])
  lengths[1:] -= depth[:-1]
  # convert to radians
  phi = np.deg2rad(phi)
  # downhole have unsigned dip values which are shorthand for negative
  if downhole:
    theta *= -1

  # in spherical coordinates theta is measured from zenith down 
  # you are measuring it from horizontal plane up 
  theta = np.deg2rad(90. - theta)

  # get x, y, z from known formulae
  x = lengths*np.sin(phi)*np.sin(theta)
  y = lengths*np.cos(phi)*np.sin(theta)
  z = lengths*np.cos(theta)

  if matrix:
    return np.column_stack((depth, x, y, z))
  else:
    # np.cumsum is employed to gradually sum resultant vectors 
    return np.column_stack((depth, np.cumsum(x), np.cumsum(y), np.cumsum(z)))

class Drillhole(object):
  _collar = None
  _survey = None
  _assay = None
  _xyz = None
  def __init__(self, collar = None, survey = None):
    super().__init__()
    if collar is None:
      collar = np.zeros(3)

    self._collar = np.array(collar)

    if survey is not None:
      self._survey = np.array(survey)
      self._xyz = np.add(desurvey_hole(self._survey[:,0],self._survey[:,1],self._survey[:,2]), [0] + self._collar.tolist())

  def getxyz(self, along = None):
    if along is None:
      return self._xyz
    v1 = np.searchsorted(self._xyz[:, 0], along)
    v0 = v1 - 1
    xyz0 = None
    xyz1 = None


    # overshoot special case
    if v1 >= self._xyz.shape[0]:
      v1 = self._xyz.shape[0] - 1
      # single interval special case
      #if self._xyz.shape[0] == 1:
      #  xyz1 = xyz0
      #else:
      #  xyz0 = self._xyz[-2]
      #  xyz1 = self._xyz[-1]
    #else:
    if v0 >= v1:
      v0 = v1 - 1

    if v0 < 0:
      xyz0 = [0] + self._collar.tolist()
    else:
      xyz0 = self._xyz[v0]

    xyz1 = self._xyz[v1]

    xyz = xyz0
    d01 = None
    t01 = None
    if v0 != v1:
      d01 = np.subtract(xyz1, xyz0)
      if np.any(d01):
        t01 = (along - xyz0[0]) / d01[0]
        xyz = np.add(xyz0, np.multiply(d01, t01))

    #print(xyz0, xyz1, along, xyz)
    #print("shape",self._xyz.shape[0],"v0",v0,"v1",v1,"d01",d01,"t01",t01,"xyz",xyz)
    return xyz
    
  def desurvey(self, table, vfrom = 'from', vto = 'to'):
    df = pd.DataFrame(table)
    df['mid_x'] = np.nan
    df['mid_y'] = np.nan
    df['mid_z'] = np.nan
    for i,row in df.iterrows():
      midxyz = self.getxyz((row[vfrom] + row[vto]) / 2)
      df.loc[i, ['mid_x','mid_y','mid_z']] = midxyz[1:]
    return df


def pd_plot_hole(df_header, df_survey, df_assay, output_img = None):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  ds = pd.DataFrame(desurvey_hole(df_survey['DEPTH'], df_survey['AZIMUTH'], df_survey['DIP']), columns=['DEPTH','X','Y','Z'])
  dh = df_header.copy()

  dxyz = dict()
  for row in dh.index:
    hid = dh.loc[row, 'HID']
    dxyz[hid] = [[0] + dh.loc[row, ['X','Y','Z']].tolist()]
    dxyz[hid] = np.vstack((dxyz[hid], [dh.loc[row, 'DEPTH'], dh.loc[row, 'X'] + ds.loc[row, 'X'], dh.loc[row, 'Y'] + ds.loc[row, 'Y'], dh.loc[row, 'Z'] + ds.loc[row, 'Z']]))

  fig = plt.figure()

  ax = plt.subplot(131, projection='3d', azim=30, elev=15)
  for k,v in dxyz.items():
    ax.plot(v.T[1], v.T[2], v.T[3], label=k)

  plt.legend()

  ax = plt.subplot(132, projection='3d', azim=120, elev=15)
  for k,v in dxyz.items():
    ax.plot(v.T[1], v.T[2], v.T[3], label=k)

  plt.legend()

  ax = plt.subplot(133, projection='3d', azim=0, elev=0)
  for k,v in dxyz.items():
    ax.plot(v.T[1], v.T[2], v.T[3], label=k)

  plt.legend()

  plt.show()
