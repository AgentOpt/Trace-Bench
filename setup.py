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

# Optional dependencies for external trainers in trace_bench/trainers/.
# Install a group with: pip install trace-bench[<key>]
# Install everything:   pip install trace-bench[all-external]
extras_require: dict = {
    # External trainers (trace_bench/trainers/)
    # "skeleton": ["skeleton-lib>=x.y"],  # template — replace with real packages

    # HuggingFace QA benchmark suite (benchmarks/hf_qa/)
    "hf": ["datasets>=2.0"],

    # External trainer adapters (trace_bench/trainers/)
    "dspy": ["dspy-ai>=2.0"],
}
extras_require["all-external"] = sorted(
    {pkg for pkgs in extras_require.values() for pkg in pkgs}
)

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
    extras_require=extras_require,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "trace-bench=trace_bench.cli:main",
        ]
    },
)
