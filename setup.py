import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_transformation",
    version="0.0.2",
    author="AInnovation",
    author_email="xiangqing@ainnovation.com",
    description="A tool to process image data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ainnovation.com/industrialvisiontang/data_transform",
    # packages=setuptools.find_packages("data_transformation"),
    packages=["data_transformation"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",

        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",

    ],
    python_requires='>=3.6',
)
