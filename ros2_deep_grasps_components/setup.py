from setuptools import setup

package_name = 'ros2_deep_grasps_components'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     #('share/' + package_name + '/config', ['config/camera_calibration.yaml']),
   ],
 install_requires=['setuptools', 'onnxruntime'],
 zip_safe=True,
 maintainer='Peter David Fagan',
 maintainer_email='peterdavidfagan@gmail.com',
 description='TODO: Package description',
 license='TODO: License declaration',
 tests_require=['pytest'],
)
