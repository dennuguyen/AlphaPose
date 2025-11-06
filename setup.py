import os
import platform
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy as np

def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != "Windows":
        extra_compile_args = {'cxx': ['-Wno-unused-function', '-Wno-write-strings']}

    extension = Extension(
        "{}.{}".format(module, name),
        [os.path.join(*module.split("."), p) for p in sources],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    extension, = cythonize(extension)
    return extension

def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )

def get_ext_modules():
    ext_modules = []
    force_compile = False
    if platform.system() != "Windows" or force_compile:
        ext_modules = [
            make_cython_ext("soft_nms_cpu", "detector.nms", ["src/soft_nms_cpu.pyx"]),
            # make_cuda_ext("nms_cpu", "detector.nms", ["src/nms_cpu.cpp"]),
            # make_cuda_ext("nms_cuda", "detector.nms", ["src/nms_cuda.cpp", "src/nms_kernel.cu"]),
            # make_cuda_ext(
            #     "roi_align_cuda",
            #     "alphapose.utils.roi_align",
            #     ["src/roi_align_cuda.cpp", "src/roi_align_kernel.cu"],
            # ),
            # make_cuda_ext(
            #     "deform_conv_cuda",
            #     "alphapose.models.layers.dcn",
            #     ["src/deform_conv_cuda.cpp", "src/deform_conv_cuda_kernel.cu"],
            # ),
            # make_cuda_ext(
            #     "deform_pool_cuda",
            #     "alphapose.models.layers.dcn",
            #     ["src/deform_pool_cuda.cpp", "src/deform_pool_cuda_kernel.cu"],
            # ),
        ]
    return ext_modules

def get_install_requires():
    install_requires = [
        "six",
        "terminaltables",
        "scipy",
        "opencv-python",
        "matplotlib",
        "visdom",
        "tqdm",
        "tensorboardx",
        "easydict",
        "pyyaml",
        "halpecocotools",
        "torch>=1.1.0",
        "torchvision>=0.3.0",
        "munkres",
        "timm==0.1.20",
        "natsort",
    ]
    if platform.system() != "Windows":
        install_requires.append("pycocotools")
    return install_requires

def is_installed(package_name):
    import pkg_resources
    return any(package_name in p.egg_name() for p in pkg_resources.working_set)

if __name__ == "__main__":
    setup(
        name="alphapose",
        version="0.6.0",
        description="Code for AlphaPose",
        long_description=readme(),
        long_description_content_type="text/markdown",
        keywords="computer vision, human pose estimation",
        url="https://github.com/MVIG-SJTU/AlphaPose",
        packages=find_packages(exclude=("data", "exp")),
        package_data={"": ["*.json", "*.txt"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        license="GPLv3",
        python_requires=">=3.8",
        setup_requires=["pytest-runner", "cython", "numpy"],
        tests_require=["pytest"],
        install_requires=get_install_requires(),
        ext_modules=get_ext_modules(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )

    # Optional: install Windows-only dependencies
    if platform.system() == "Windows" and not is_installed("pycocotools"):
        os.system("python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI")
    if not is_installed("cython_bbox"):
        os.system("python -m pip install git+https://github.com/yanfengliu/cython_bbox.git")
