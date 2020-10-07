import setuptools
import versioneer

setuptools.setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IDConn",  # Replace with your own username
    version="0.2dev",
    author="Katie Bottenhorn",
    author_email="katie.bottenhorn@gmail.com",
    description="A Python pipeline for studying individual differences in brain connectivity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NBCLab/IDConn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
