from setuptools import find_packages, setup

package_name = "my_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/test", ["test/test_pose_integration.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Proximity-first cable insertion policy — Sprint 1",
    license="Apache-2.0",
    extras_require={
        "test": ["pytest"],
    },
)
