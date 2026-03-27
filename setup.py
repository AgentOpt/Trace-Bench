import setuptools

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
    version="0.4.0",
    author="Trace Team",
    author_email="chinganc0@gmail.com, aimingnie@gmail.com, adith387@gmail",
    url="https://github.com/AgentOpt/Trace-Bench",
    license="MIT",
    description="Benchmarking framework for AI optimization algorithms built on OpenTrace",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["trace_bench*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "trace-bench=trace_bench.cli:main",
        ]
    },
)
