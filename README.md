# FDTD-PyCUDA

A Finite Difference Time Domain (FDTD) electromagnetic simulation project with CUDA acceleration and AI-based boundary condition handling.

## Overview

This project implements FDTD simulations for electromagnetic wave propagation in 1D, 2D, and 3D domains. It includes:

- **CPU implementations** for baseline performance
- **CUDA-accelerated implementations** using PyCUDA for GPU computing
- **Perfectly Matched Layer (PML)** boundary condition implementations
- **AI-based boundary condition handling** using neural networks (replacing traditional PML)

## Project Structure

```
FDTD-PyCuda/
├── A dimension/          # 1D FDTD simulations
│   ├── CPU_code.py       # CPU implementation
│   ├── CUDA_code.py      # CUDA implementation
│   └── tempos_*          # Performance timing files
├── Two dimensions/       # 2D FDTD simulations
│   ├── CPU_code.py       # CPU implementation with PML
│   ├── CUDA_code.py      # CUDA implementation with AI boundaries
│   ├── rede.ipynb        # Neural network training notebook
│   ├── Imagens/          # Visualization outputs
│   └── *.npy             # Simulation result arrays
└── Three dimensions/     # 3D FDTD simulations
    ├── CPU_code.py       # CPU implementation
    ├── CUDA_code.py      # CUDA implementation
    └── *.png             # Visualization outputs
```

## Features

### 1D Simulations
- Basic FDTD wave propagation
- Gaussian pulse source
- Performance benchmarking

### 2D Simulations
- 2D FDTD with PML boundary conditions
- CUDA acceleration for field updates
- Neural network-based boundary absorption
- Visualization of electric field (Ez) distribution

### 3D Simulations
- Full 3D FDTD implementation
- Six field components (Ex, Ey, Ez, Hx, Hy, Hz)
- Dipole antenna simulation support
- 3D visualization outputs

## Requirements

### Dependencies
- Python 3.x
- NumPy
- Matplotlib
- PyCUDA
- Keras/TensorFlow (for AI-based boundary conditions)
- tqdm (progress bars)

### Hardware Requirements
- NVIDIA GPU with CUDA support (for CUDA implementations)
- CUDA Toolkit installed

### Installation

```bash
pip install numpy matplotlib pycuda keras tensorflow tqdm
```

## Usage

### Running CPU Simulations

**1D:**
```bash
cd "A dimension"
python CPU_code.py
```

**2D:**
```bash
cd "Two dimensions"
python CPU_code.py
```

**3D:**
```bash
cd "Three dimensions"
python CPU_code.py
```

### Running CUDA Simulations

**1D:**
```bash
cd "A dimension"
python CUDA_code.py
```

**2D (with AI boundaries):**
```bash
cd "Two dimensions"
python CUDA_code.py
```
*Note: Requires trained model file `pml-IA`*

**3D:**
```bash
cd "Three dimensions"
python CUDA_code.py
```

## Key Implementation Details

### FDTD Algorithm
The FDTD method discretizes Maxwell's equations in time and space:
- **E-field updates**: Based on H-field curl
- **H-field updates**: Based on E-field curl
- **Leapfrog time stepping**: E and H fields are updated alternately

### Boundary Conditions

**PML (Perfectly Matched Layer):**
- Traditional absorbing boundary conditions
- Polynomial absorption profile
- 8-layer PML implementation

**AI-Based Boundaries:**
- Neural network predicts boundary field values
- Trained to mimic PML behavior
- Applied to boundary regions during simulation

### CUDA Kernels
- `fdtd_e`: Updates electric field components on GPU
- `fdtd_h`: Updates magnetic field components on GPU
- Block and grid dimensions optimized for GPU architecture

## Performance Benchmarks

The project includes timing files (`tempos_*`) comparing:
- CPU vs CUDA execution times
- PML vs AI-based boundary conditions
- Scaling with grid size

Results are saved as:
- Text files: `tempos_cpu`, `tempos_cuda_pml`, `tempos_cuda_ia`
- Excel files: `Tempos.xlsx`, `Tempos_1d.xlsx`, `Tempos_3d.xlsx`

## Output Files

### Data Files
- `*.npy`: NumPy arrays containing simulation results (field values over time)
- `tempos_*`: Performance timing data

### Visualization Files
- `img-*.png`: 2D/3D plots of electromagnetic fields
- `pml_*.png`: PML boundary visualization
- `ia_*.png`: AI boundary implementation results

## Notes

- Grid sizes are configurable in each script
- Default simulation uses 60×60 (2D) or 60×60×60 (3D) grids
- Time step size is automatically calculated based on cell size and Courant condition
- Gaussian and sinusoidal pulse sources are implemented
- Results are saved automatically for post-processing

## License

[Add your license here]

## Authors

[Add author information here]

