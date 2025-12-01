from setuptools import setup, find_packages

setup(
    name="observer-mask-simulation",
    version="1.0.0",
    author="Haley Gillett",
    author_email="",
    description="Simulation of inter-observer variability for brain tumor segmentation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    py_modules=["simulate_observers"],
    install_requires=[
        "numpy>=1.20",
        "SimpleITK>=2.0",
        "scipy>=1.6",
    ],
    entry_points={
        "console_scripts": [
            "simulate-observers=simulate_observers:main",
        ]
    },
    python_requires=">=3.8",
)
