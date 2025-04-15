from setuptools import setup, find_packages

setup(
    name="voice-emotion-detector",
    version="0.1.0",
    description="A containerized voice emotion detection application.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yifan Zhang, Shuyuan Yang, Eli Sun, Jasmine Zhang",
    url="https://github.com/software-students-spring2025/4-containers-emmmm",
    packages=find_packages(),  # This should detect the web_app package if structured properly.
    install_requires=[
        "flask==3.1.0",
        "requests==2.32.3",
        "pymongo==4.12.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
