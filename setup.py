"""
Setup script for Face Mood Music Player.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="facemood-music-player",
    version="2.0.0",
    description="Layered affect-aware music: perception → temporal fusion → policy → audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Face Mood Music Player Team",
    url="https://github.com/iceman404/facemood_music_player",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "facemood-player=facemood.app.application:main",
            "facemood-gui=facemood.gui.app:run_gui",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Players",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)

