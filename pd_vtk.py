#!python
'''
Copyright 2017 - 2021 Vale

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import numpy as np
import pandas as pd
try:
  import pyvista as pv
except:
  # we use this base class in enviroments that dont support VTK
  class pv(object):
    class UniformGrid(object):
      pass
    def read(*argv):
      pass
    

''' GetDataObjectType
PolyData == 0
VTK_STRUCTURED_GRID = 2
VTK_RECTILINEAR_GRID = 3
VTK_UNSTRUCTURED_GRID = 4
UniformGrid == 6
VTK_MULTIBLOCK_DATA_SET = 13
'''

def pv_read(fp):
  ''' simple import safe pyvista reader '''
  if pv is None: return
  return pv.read(fp)

def pv_save(meshes, fp, binary=True):
  ''' simple import safe pyvista writer '''
  if pv is None: return
  if not isinstance(meshes, list):
    meshes.save(fp, binary)
  elif len(meshes) == 1:
    meshes[0].save(fp, binary)
  else:
    pv.MultiBlock(meshes).save(fp, binary)


def vtk_cells_to_flat(cells):
  r = []
  p = 0
  while p < len(cells):
    n = cells[p]
    r.extend(cells[p+1:p+1+n])
    p += n + 1
  return r

def vtk_cells_to_faces(cells):
  faces = vtk_cells_to_flat(cells)
  return np.reshape(faces, (len(faces) // 3, 3))

def vtk_flat_to_cells(flat, nodes = None):
  #print(flat)
  #print(nodes)
  if nodes is None:
    nodes = pd.Series(np.arange(len(flat)), flat.index)
  n = 0
  cells = []
  for i in flat.index[::-1]:
    n += 1
    cells.insert(0, nodes[i])
    if flat[i] == 0:
      cells.insert(0, n)
      n = 0
  return np.array(cells)

def pd_detect_xyz(df, z = True):
  xyz = None
  dfcs = set(df.columns)
  for s in [['x','y','z'], ['mid_x','mid_y','mid_z'], ['xworld','yworld','zworld'], ['xc','yc','zc']]:
    if z == False:
      s.pop()
    for c in [str.lower, str.upper,str.capitalize]:
      cs = list(map(c, s))
      if dfcs.issuperset(cs):
        xyz = cs
        break
    else:
      continue
    # break also the outter loop if the inner loop ended due to a break
    break
  if xyz is None and z:
    return pd_detect_xyz(df, False)
  return xyz

def vtk_nf_to_mesh(nodes, faces):
  if len(nodes) == 0:
    return pv.PolyData()
  if len(faces) == 0:
    return pv.PolyData(np.array(nodes))
  meshfaces = np.hstack(np.concatenate((np.full((len(faces), 1), 3, dtype=np.int_), faces), 1))
  return pv.PolyData(np.array(nodes), meshfaces)

def vtk_df_to_mesh(df, xyz = None):
  if pv is None: return
  if xyz is None:
    xyz = pd_detect_xyz(df)
  if xyz is None:
    print('geometry/xyz information not found')
    return None
  print("xyz:",','.join(xyz))
  if len(xyz) == 2:
    xyz.append('z')
    if 'z' not in df:
      if '0' in df:
        # geotiff first/only spectral channel
        print('using first channel as Z value')
        df['z'] = df['0']
      else:
        print('using 0 as Z value')
        df['z'] = 0

  #pdata = df[xyz].dropna(0, 'all')
  #pdata.fillna(0, inplace=True)
  pdata = df[xyz]

  if 'n' in df and df['n'].max() > 0:
    if 'node' in df:
      cells = vtk_flat_to_cells(df['n'], df['node'])
      nodes = df['node'].drop_duplicates().sort_values()
      pdata = pdata.loc[nodes.index]
    else:
      cells = vtk_flat_to_cells(df['n'])

    mesh = pv.PolyData(pdata.values.astype(np.float), cells)
  else:
    mesh = pv.PolyData(pdata.values.astype(np.float))
  if 'colour' in df:
    mesh.point_arrays['colour'] = df.loc[pdata.index, 'colour']
  return mesh

# dmbm_to_vtk
def vtk_dmbm_to_ug(df):
  ''' datamine block model to uniform grid '''
  df_min = df.min(0)
  xyzc = ['XC','YC','ZC']

  size = df_min[['XINC','YINC','ZINC']].astype(np.int_)

  dims = np.add(df_min[['NX','NY','NZ']] ,1).astype(np.int_)

  origin = df_min[['XMORIG','YMORIG','ZMORIG']]

  grid = pv.UniformGrid(dims, size, origin)
  n_predefined = 13
  vl = [df.columns[_] for _ in range(13, df.shape[1])]
  
  cv = [dict()] * grid.GetNumberOfCells()

  for i,row in df.iterrows():
    cell = grid.find_closest_cell(row[xyzc].values)
    if cell >= 0:
      cv[cell] = row[vl].to_dict()
  cvdf = pd.DataFrame.from_records(cv)
  for v in vl:
    grid.cell_arrays[v] = cvdf[v]

  return grid

def vtk_plot_meshes(meshes, point_labels=False):
  if pv is None: return
  p = pv.Plotter()
  c = 0
  for mesh in meshes:
    if mesh is not None:
      p.add_mesh(mesh, opacity=0.5)
      if point_labels:
        p.add_point_labels(mesh.points, np.arange(mesh.n_points))
      c += 1
  if c:
    print("display", c, "meshes")
    p.show()

def vtk_mesh_to_df(mesh):
  df = pd.DataFrame()
  if hasattr(mesh, 'n_blocks'):
    for block in mesh:
      df = df.append(vtk_mesh_to_df(block))
  else:
    arr_n = np.zeros(mesh.n_points, dtype=np.int)
    arr_node = np.arange(mesh.n_points, dtype=np.int)
    print("GetDataObjectType", mesh.GetDataObjectType())
    # VTK_STRUCTURED_POINTS = 1
    # VTK_STRUCTURED_GRID = 2
    # VTK_UNSTRUCTURED_GRID = 4
    # 6 = UniformGrid
    # VTK_UNIFORM_GRID = 10
    if mesh.GetDataObjectType() in [2,4,6]:
      points = mesh.cell_centers().points
      arr_node = np.arange(mesh.GetNumberOfCells(), dtype=np.int)
      arr_n = np.zeros(mesh.GetNumberOfCells())
      arr_data = [pd.Series(mesh.get_array(name),name=name) for name in mesh.cell_arrays]
    else:
      arr_data = []
      # in some cases, n_faces may be > 0  but with a empty faces array
      if mesh.n_faces and len(mesh.faces):
        face_size = int(mesh.faces[0])
        arr_flat = vtk_cells_to_flat(mesh.faces)
        points = mesh.points.take(arr_flat, 0)
        arr_node = arr_node.take(arr_flat)
        arr_n = np.tile(np.arange(face_size, dtype=np.int), len(points) // face_size)
        for name in mesh.point_arrays:
          arr_data.append(pd.Series(mesh.get_array(name).take(arr_flat), name=name))
      else:
        points = mesh.points
        arr_data = [pd.Series(mesh.point_arrays[name],name=name) for name in mesh.point_arrays]
   
    df = pd.concat([pd.DataFrame(points,columns=['x','y','z']), pd.Series(arr_n,name='n'), pd.Series(arr_node,name='node')] + arr_data,1)

  return df

def vtk_mesh_info(mesh):
  print(mesh)
  #.IsA('vtkMultiBlockDataSet'):
  if hasattr(mesh, 'n_blocks'):
    for n in range(mesh.n_blocks):
      print("block",n,"name",mesh.get_block_name(n))
      vtk_mesh_info(mesh.get(n))
  else:
    for preference in ['point', 'cell', 'field']:
      arr_list = mesh.cell_arrays
      if preference == 'point':
        arr_list = mesh.point_arrays
      if preference == 'field':
        arr_list = mesh.field_arrays

      for name in arr_list:
        arr = mesh.get_array(name, preference)
        # check if this array is unicode, obj, str or other text types
        if arr.dtype.num >= 17:
          d = np.unique(arr)
        else:
          d = '{%f <=> %f}' % mesh.get_data_range(name, preference)
        print(name,preference,arr.dtype.name,d,len(arr))
    print('')
  return mesh

def vtk_array_string_to_index(mesh):
  print("converting string arrays to integer index:")
  for name in mesh.cell_arrays:
    arr = mesh.cell_arrays[name]
    if arr.dtype.num >= 17:
      print(name,"(cell)",arr.dtype)
      mesh.cell_arrays[name] = pd.factorize(arr)[0]
  for name in mesh.point_arrays:
    arr = mesh.point_arrays[name]
    if arr.dtype.num >= 17:
      print(name,"(point)",arr.dtype)
      mesh.point_arrays[name] = pd.factorize(arr)[0]
  return mesh

def vtk_info(fp):
  if pv is None: return
  return vtk_mesh_info(pv.read(fp))


class vtk_Voxel(pv.UniformGrid):
  @classmethod
  def from_bmf(cls, bm, n_schema = None):
    if n_schema is None:
      n_schema = bm.model_n_schemas()-1
    size = np.resize(bm.model_schema_size(n_schema), 3)
    dims = bm.model_schema_dimensions(n_schema)
    o0 = bm.model_schema_extent(n_schema)
    origin = np.add(bm.model_origin(), o0[:3])
    return cls(np.add(dims, 1, dtype = np.int_, casting = 'unsafe'), size, origin[:3])

  @classmethod
  def from_mesh(cls, mesh, cell_size = 10, ndim = 3):
    mesh = mesh.copy()
    if ndim == 2:
      mesh.points[:, 2] = 0

    bb = np.transpose(np.reshape(mesh.GetBounds(), (3,2)))
    
    dims = np.add(np.ceil(np.divide(np.subtract(bb[1], bb[0]), cell_size)), 3)
    if ndim == 2:
      dims[2] = 1
    origin = np.subtract(bb[0], cell_size)
    #grid = pv.UniformGrid(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)
    return cls(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)

  @classmethod
  def from_df(cls, df, cell_size, xyz = ['x','y','z']):

    bb0 = df[xyz].min()
    bb1 = df[xyz].max()
    dims = np.add(np.ceil(np.divide(np.subtract(bb1, bb0), cell_size)), 3)
    origin = np.subtract(bb0, cell_size)

    return cls(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)

  @property
  def shape(self):
    shape = np.subtract(self.dimensions, 1)
    return shape[shape.nonzero()]

  def get_ndarray(self, name = None, preference='cell'):
    if name is None:
      return np.ndarray(self.shape)
    return self.get_array(name, preference).reshape(self.shape)

  def set_ndarray(self, name, array, preference='cell'):
    if preference=='cell':
      self.cell_arrays[name] = array.flat
    else:
      self.point_arrays[name] = array.flat

  def GetCellCenter(self, cellId):
    return vtk_Voxel.sGetCellCenter(self, cellId)

  # DEPRECATED: use cell_centers().points
  @staticmethod
  def sGetCellCenter(self, cellId):
    cell = self.GetCell(cellId)
    bounds = np.reshape(cell.GetBounds(), (3,2))
    return bounds.mean(1)

def vtk_texture_to_array(tex):
  ' WORKING drop in replacement for to_array()'
  img = tex.to_image()
  sh = (img.dimensions[1], img.dimensions[0])
  if img.active_scalars.ndim > 1:
    sh = (img.dimensions[1], img.dimensions[0], tex.n_components)
  return img.active_scalars.reshape(sh)

def vtk_to_gltf(vtk_meshes):
  import pygltflib
  from pygltflib.utils import ImageFormat
  import PIL.Image
  import skimage.io
  import io
  buffer0 = io.BytesIO()
  accessors = []
  bufferviews = []
  meshes = []
  texcoords = []
  textures = []
  images = []
  samplers = []
  materials = []
  for mesh in vtk_meshes:
    nodes = mesh.points
    faces = vtk_cells_to_faces(mesh.faces)
    tcoor = mesh.t_coords
    meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=len(accessors),TEXCOORD_0=len(texcoords)+2), indices=len(accessors)+1, material=0)]))
    # POSITION
    view_blob = nodes.astype(np.float32).tobytes()
    bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ARRAY_BUFFER)
    accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.FLOAT,count=len(nodes),type=pygltflib.VEC3,max=nodes.max(axis=0).tolist(),min=nodes.min(axis=0).tolist())
    buffer0.write(view_blob)
    bufferviews.append(bufferview)
    accessors.append(accessor)
    # indices
    view_blob = faces.astype(np.int).tobytes()
    bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ELEMENT_ARRAY_BUFFER)
    accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.UNSIGNED_INT,count=faces.size,type=pygltflib.SCALAR,max=[],min=[])
    buffer0.write(view_blob)
    bufferviews.append(bufferview)
    accessors.append(accessor)
    # TEXCOORD
    view_blob = tcoor.astype(np.float32).tobytes()
    bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ARRAY_BUFFER)
    accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.FLOAT,count=len(tcoor),type=pygltflib.VEC2,max=[],min=[])
    buffer0.write(view_blob)
    bufferviews.append(bufferview)
    accessors.append(accessor)
    for i in mesh.textures:
      byteoffset = buffer0.tell()
      img = vtk_texture_to_array(mesh.textures[i])
      skimage.io.imsave(buffer0, img, format='png')
      # buffers chunks MUST be multiple of 4
      while buffer0.tell() % 4 > 0:
        buffer0.write(b'\0')
      materials.append(pygltflib.Material(doubleSided=True, alphaCutoff=None, pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorTexture=pygltflib.TextureInfo(index=len(textures), texCoord=0))))
      textures.append(pygltflib.Texture(source=len(images)))
      images.append(pygltflib.Image(mimeType=pygltflib.IMAGEPNG,bufferView=len(bufferviews)))
      bufferviews.append(pygltflib.BufferView(buffer=0,byteOffset=byteoffset,byteLength=buffer0.tell()-byteoffset))

  gltf = pygltflib.GLTF2(
      scene=0,
      scenes=[pygltflib.Scene(nodes=[0])],
      nodes=[pygltflib.Node(mesh=0)],
      meshes=meshes,
      accessors=accessors,
      bufferViews=bufferviews,
      buffers=[
          pygltflib.Buffer(byteLength=buffer0.tell())
      ],
      images=images,
      samplers=samplers,
      textures=textures,
      materials=materials
  )

  gltf.set_binary_blob(buffer0.getbuffer())

  print(gltf)

  return gltf

if __name__=="__main__":
  pass