import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvdata",
    version="0.0.2",
    author="XiangQing",
    author_email="...",
    description="A tool to process cv data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThistleInTheSun/CV_Data_Prcoessing",
    packages=["cvdata"],
    python_requires='>=3.6',
)
