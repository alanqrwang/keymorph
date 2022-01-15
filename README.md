# KeypointMorph: Robust Multi-modal Affine Registration via Unsupervised Keypoint Detection

Registration is a fundamental task in medical imaging, and recent machine learning methods have become the state-of-the-art. However, these approaches are often not interpretable, lack robustness to large misalignments, and do not incorporate symmetries of the problem. In this work, we propose KeypointMorph, an unsupervised end-to-end learning-based image registration framework that relies on automatically detecting corresponding keypoints. Our core insight is straightforward: matching keypoints between images can be used to obtain the optimal transformation via a differentiable closed-form expression. We use this observation to drive the unsupervised learning of anatomically-consistent keypoints from images. This not only leads to substantially more robust registration but also yields better interpretability, since the keypoints reveal which parts of the image are driving the final alignment. Moreover, KeypointMorph can be designed to be equivariant under image translations and/or symmetric with respect to the input image ordering. We demonstrate the proposed framework in solving 3D affine registration of multi-modal brain MRI scans. Remarkably, we show that this strategy leads to consistent keypoints, even across modalities. We demonstrate registration accuracy that surpasses current state-of-the-art methods, especially in the context of large displacements.

## Requirements
We tested our code in Python 3.8 and the following packages:
- pytorch 1.10
- torchvision 1.11
- jupyterlab 3.2.1
- ipywidgets 7.6.5
- simpleitk 2.1.1
- scikit-image 0.18.3
- scikit-learn 1.0.1
- seaborn 0.11.2
- tqdm 4.62.3
- torchio 0.18.62
- numpy 1.21.3
- h5py 3.5.0
- antspyx 0.3.1

In the demo `./notebook/[B] Skullstripping.ipynb`, we also use theano 1.0.5 for skullstrip
