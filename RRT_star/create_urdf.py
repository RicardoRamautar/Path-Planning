import os
import json
from variables import RAD_SPHERE

def create_urdf_file(point_positions, urdf_filename):
    # urdf_data = '<?xml version="1.0" ?>\n'
    urdf_data  = '<robot name="simple_ball">\n'
    urdf_data += '  <link name="ball">\n'
    urdf_data += '    <inertial>\n'
    urdf_data += '      <mass value="0.0" />\n'
    urdf_data += '      <origin xyz="0 0 0" />\n'
    urdf_data += '      <inertia  ixx="1.0" ixy="0.0"  ixz="0.0"  iyy="1.0"  iyz="0.0"  izz="1.0" />\n'
    urdf_data += '    </inertial>\n'

    for i, position in enumerate(point_positions):
        x, y, z = position
        urdf_data += f'    <visual>\n'
        urdf_data += f'      <origin xyz="{x} {y} {z}" rpy="0 0 0" />\n'
        urdf_data += f'      <geometry>\n'
        urdf_data += f'        <sphere radius="{RAD_SPHERE}" />\n'
        urdf_data += f'      </geometry>\n'
        urdf_data += f'    </visual>\n'

        urdf_data += f'    <collision>\n'
        urdf_data += f'      <origin xyz="{x} {y} {z}" rpy="0 0 0" />\n'
        urdf_data += f'      <geometry>\n'
        urdf_data += f'        <sphere radius="{RAD_SPHERE}" />\n'
        urdf_data += f'      </geometry>\n'
        urdf_data += f'    </collision>\n'

    urdf_data += '    <collision>\n'
    urdf_data += '      <origin xyz="0 0 0" rpy="0 0 0" />\n'
    urdf_data += '      <geometry>\n'
    urdf_data += '        <sphere radius="{RAD_SPHERE}" />\n'
    urdf_data += '      </geometry>\n'
    urdf_data += '    </collision>\n'
    urdf_data += '  </link>\n'
    urdf_data += '  <gazebo reference="ball">\n'
    urdf_data += '    <mu1>10</mu1>\n'
    urdf_data += '    <mu2>10</mu2>\n'
    urdf_data += '    <material>Gazebo/Red</material>\n'
    urdf_data += '    <turnGravityOff>false</turnGravityOff>\n'
    urdf_data += '  </gazebo>\n'
    urdf_data += '</robot>'

    with open(urdf_filename, 'w') as urdf_file:
        urdf_file.write(urdf_data)

script_dir = os.path.dirname(os.path.abspath(__file__))
spheres_filename = os.path.join(script_dir, 'maze.json')
urdf_filename = os.path.join(script_dir, 'spheres_maze.urdf')

f = open(spheres_filename)
data = json.load(f)
sphere_positions = data['obstacles']

create_urdf_file(sphere_positions, urdf_filename)
print(f'URDF file "{urdf_filename}" created.')