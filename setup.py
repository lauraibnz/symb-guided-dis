from setuptools import setup, find_packages

setup(
    name="symb-guided-dis",
    version="0.1.0",
    description="Symbolic-guided music disentanglement for controllable generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"symb_guided_dis.training": ["configs/*.gin"]},
    python_requires=">=3.11,<3.12",
)
