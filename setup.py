import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
__version__ = "0.1.0"

install_requires = [
    "graphviz>=0.20.1",
    "pytest",
    "litellm==1.75.0",
    "aiohttp>=3.9,<3.13",
    "black",
    "scikit-learn",
    "tensorboardX",
    "tensorboard",
    "pyyaml",
]

setuptools.setup(
    name="trace-bench",
    version=__version__,
    author="Trace Team",
    author_email="chinganc0@gmail.com, aimingnie@gmail.com, adith387@gmail",
    url="https://github.com/AgentOpt/Trace",
    license='MIT LICENSE',
    description="An AutoDiff-like tool for training AI systems end-to-end with general feedback",
    long_description=open('README.md', encoding="utf8").read(),
    packages=setuptools.find_packages(include=["trace_bench*", "opto*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "trace-bench=trace_bench.cli:main",
        ]
    },
)
