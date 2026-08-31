from setuptools import find_packages, setup

package_name = 'bigreddog_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='spot',
    maintainer_email='louisl0517l@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'depth_viewer = bigreddog_controller.value_check:main',
            'depth_encoder_tester = bigreddog_controller.controller:main',
        ],
    },
)
