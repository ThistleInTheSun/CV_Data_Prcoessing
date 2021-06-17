import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_transforms",
    version="0.0.1a",
    author="AInnovation",
    author_email="xiangqing@ainnovation.com",
    description="A tool to simulate defects in industrial quality inspection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ainnovation.com/manuvision",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",

        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",

    ],
    python_requires='>=3.6',
)
