import streamlit as st
import os
import tempfile
import torch
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.visualize import view
import py3Dmol
from mace.calculators import mace_mp
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import pandas as pd

from huggingface_hub import login

try:
    hf_token = st.secrets["HF_TOKEN"]["token"]
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
except Exception as e:
    print("streamlit hf secret not defined/assigned")

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Check if running on Streamlit Cloud vs locally
is_streamlit_cloud = os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud'
MAX_ATOMS_CLOUD = 50  # Maximum atoms allowed on Streamlit Cloud
MAX_ATOMS_CLOUD_UMA = 15

# Set page configuration
st.set_page_config(
    page_title="Molecular Structure Analysis",
    page_icon="🧪",
    layout="wide"
)

# Add CSS for better formatting
# st.markdown("""
# <style>
# .stApp {
#     max-width: 1200px;
#     margin: 0 auto;
# }
# .main-header {
#     font-size: 2.5rem;
#     font-weight: bold;
#     margin-bottom: 1rem;
# }
# .section-header {
#     font-size: 1.5rem;
#     font-weight: bold;
#     margin-top: 1.5rem;
#     margin-bottom: 1rem;
# }
# .info-text {
#     font-size: 1rem;
#     color: #555;
# }
# </style>
# """, unsafe_allow_html=True)

# Title and description
st.markdown('## MLIP Playground', unsafe_allow_html=True)
st.write('#### Run atomistic simulations with state-of-the-art universal machine learning interatomic potentials (MLIPs) for molecules and materials')
st.markdown('Upload molecular structure files or select from predefined examples, then compute energies and forces using foundation models such as those from MACE or FairChem (Meta).', unsafe_allow_html=True)

# Create a directory for sample structures if it doesn't exist
SAMPLE_DIR = "sample_structures"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Dictionary of sample structures
SAMPLE_STRUCTURES = {
    "Water": "H2O.xyz",
    "Methane": "CH4.xyz",
    "Benzene": "C6H6.xyz",
    "Ethane": "C2H6.xyz",
    "Caffeine": "caffeine.xyz",
    "Ibuprofen": "ibuprofen.xyz"
}



# Custom logger that updates the table
def streamlit_log(opt):
    energy = opt.atoms.get_potential_energy()
    forces = opt.atoms.get_forces()
    fmax_step = np.max(np.linalg.norm(forces, axis=1))
    opt_log.append({
        "Step": opt.nsteps,
        "Energy (eV)": round(energy, 6),
        "Fmax (eV/Å)": round(fmax_step, 6)
    })
    df = pd.DataFrame(opt_log)
    table_placeholder.dataframe(df)

# Function to check atom count limits
def check_atom_limit(atoms_obj, selected_model):
    if atoms_obj is None:
        return True
    
    num_atoms = len(atoms_obj)
    if ('UMA' in selected_model or 'ESEN MD' in selected_model) and num_atoms > MAX_ATOMS_CLOUD_UMA:
        st.error(f"⚠️ Error: Your structure contains {num_atoms} atoms, which exceeds the {MAX_ATOMS_CLOUD_UMA} atom limit for Streamlit Cloud deployments for large sized FairChem models. For larger systems, please download the repository from GitHub and run it locally on your machine where no atom limit applies.")
        st.info("💡 Running locally allows you to process much larger structures and use your own computational resources more efficiently.")
        return False
    if num_atoms > MAX_ATOMS_CLOUD:
        st.error(f"⚠️ Error: Your structure contains {num_atoms} atoms, which exceeds the {MAX_ATOMS_CLOUD} atom limit for Streamlit Cloud deployments. For larger systems, please download the repository from GitHub and run it locally on your machine where no atom limit applies.")
        st.info("💡 Running locally allows you to process much larger structures and use your own computational resources more efficiently.")
        return False
    return True


# Define the available MACE models
MACE_MODELS = {
    "MACE MPA Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "MACE OMAT Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "MACE MATPES r2SCAN Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "MACE MATPES PBE Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "MACE MP 0a Small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "MACE MP 0a Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "MACE MP 0a Large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model",
    "MACE MP 0b Small": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "MACE MP 0b Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "MACE MP 0b2 Small": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "MACE MP 0b2 Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "MACE MP 0b2 Large": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "MACE MP 0b3 Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
}

# Define the available FairChem models
FAIRCHEM_MODELS = {
    "UMA Small": "uma-sm",
    "ESEN MD Direct All OMOL": "esen-md-direct-all-omol",
    "ESEN SM Conserving All OMOL": "esen-sm-conserving-all-omol",
    "ESEN SM Direct All OMOL": "esen-sm-direct-all-omol"
}

@st.cache_resource
def get_mace_model(model_path, device, selected_default_dtype):
    # Create a model of the specified type.
    return mace_mp(model=model_path, device=device, default_dtype=selected_default_dtype)

@st.cache_resource
def get_fairchem_model(selected_model, model_path, device, selected_task_type):
    predictor = pretrained_mlip.get_predict_unit(model_path, device=device)
    if selected_model == "UMA Small":
        calc = FAIRChemCalculator(predictor, task_name=selected_task_type)
    else:
        calc = FAIRChemCalculator(predictor)
    return calc

# Sidebar for file input and parameters
st.sidebar.markdown("## Input Options")

# Input method selection
input_method = st.sidebar.radio("Choose Input Method:", ["Select Example", "Upload File", "Paste Content"])

# Initialize atoms variable
atoms = None

# File upload option
if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload structure file", 
                                           type=["xyz", "cif", "POSCAR", "mol", "tmol"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        try:
            # Read the structure using ASE
            atoms = read(tmp_filepath)
            st.sidebar.success(f"Successfully loaded structure with {len(atoms)} atoms!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            
        # Clean up the temporary file
        os.unlink(tmp_filepath)

# Example structure selection
elif input_method == "Select Example":
    example_name = st.sidebar.selectbox("Select Example Structure:", list(SAMPLE_STRUCTURES.keys()))
    
    if example_name:
        file_path = os.path.join(SAMPLE_DIR, SAMPLE_STRUCTURES[example_name])
        try:
            atoms = read(file_path)
            st.sidebar.success(f"Loaded {example_name} with {len(atoms)} atoms!")
        except Exception as e:
            st.sidebar.error(f"Error loading example: {str(e)}")

# Paste content option
elif input_method == "Paste Content":
    file_format = st.sidebar.selectbox("File Format:", 
                                      ["XYZ", "CIF", "extXYZ", "POSCAR (VASP)", "Turbomole", "MOL"])
    
    content = st.sidebar.text_area("Paste file content here:", height=200)
    
    if content and st.sidebar.button("Parse Content"):
        try:
            # Create a temporary file with the pasted content
            suffix_map = {"XYZ": ".xyz", "CIF": ".cif", "extXYZ": ".extxyz", 
                         "POSCAR (VASP)": ".POSCAR", "Turbomole": ".tmol", "MOL": ".mol"}
            
            suffix = suffix_map.get(file_format, ".xyz")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(content.encode())
                tmp_filepath = tmp_file.name
            
            # Read the structure using ASE
            atoms = read(tmp_filepath)
            st.sidebar.success(f"Successfully parsed structure with {len(atoms)} atoms!")
            
            # Clean up the temporary file
            os.unlink(tmp_filepath)
        except Exception as e:
            st.sidebar.error(f"Error parsing content: {str(e)}")


# Model selection
st.sidebar.markdown("## Model Selection")
model_type = st.sidebar.radio("Select Model Type:", ["MACE", "FairChem"])

selected_task_type = None
if model_type == "MACE":
    selected_model = st.sidebar.selectbox("Select MACE Model:", list(MACE_MODELS.keys()))
    model_path = MACE_MODELS[selected_model]
    if selected_model == "MACE OMAT Medium":
        st.sidebar.warning("Using model under Academic Software License (ASL) license, see [https://github.com/gabor1/ASL](https://github.com/gabor1/ASL). To use this model you accept the terms of the license.")
    selected_default_dtype = st.sidebar.selectbox("Select Precision (default_dtype):", ['float32', 'float64'])
if model_type == "FairChem":
    selected_model = st.sidebar.selectbox("Select FairChem Model:", list(FAIRCHEM_MODELS.keys()))
    model_path = FAIRCHEM_MODELS[selected_model]
    if selected_model == "UMA Small":
        st.sidebar.warning("Meta FAIR Acceptable Use Policy. This model was developed by the Fundamental AI Research (FAIR) team at Meta. By using it, you agree to their acceptable use policy, which prohibits using their models to violate the law or others' rights, plan or develop activities that present a risk of death or harm, and deceive or mislead others.")
        selected_task_type = st.sidebar.selectbox("Select UMA Model Task Type:", ["omol", "omat", "omc", "odac", "oc20"])
# Check atom count limit
if atoms is not None:
    check_atom_limit(atoms, selected_model)
    #st.sidebar.success(f"Successfully parsed structure with {len(atoms)} atoms!")
# Device selection
device = st.sidebar.radio("Computation Device:", ["CPU", "CUDA (GPU)"], 
                         index=0 if not torch.cuda.is_available() else 1)
device = "cuda" if device == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"

if device == "cpu" and torch.cuda.is_available():
    st.sidebar.info("GPU is available but CPU was selected. Calculations will be slower.")
elif device == "cpu" and not torch.cuda.is_available():
    st.sidebar.info("No GPU detected. Using CPU for calculations.")

# Task selection
st.sidebar.markdown("## Task Selection")
task = st.sidebar.selectbox("Select Calculation Task:", 
                           ["Energy Calculation", 
                            "Energy + Forces Calculation", 
                            "Geometry Optimization", 
                            "Cell + Geometry Optimization"])

# Optimization parameters
if "Optimization" in task:
    st.sidebar.markdown("### Optimization Parameters")
    max_steps = st.sidebar.slider("Maximum Steps:", min_value=10, max_value=25, value=15, step=1)
    fmax = st.sidebar.slider("Convergence Threshold (eV/Å):", 
                            min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%.3f")
    optimizer = st.sidebar.selectbox("Optimizer:", ["BFGS", "LBFGS", "FIRE"], index=1)

# Main content area
if atoms is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### Structure Visualization', unsafe_allow_html=True)
        
        # Generate visualization
        def get_structure_viz(atoms_obj):
            # Convert atoms to XYZ format
            xyz_str = ""
            xyz_str += f"{len(atoms_obj)}\n"
            xyz_str += "Structure\n"
            for atom in atoms_obj:
                xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
            
            # Create a py3Dmol visualization
            view = py3Dmol.view(width=400, height=400)
            view.addModel(xyz_str, "xyz")
            view.setStyle({'stick': {}})
            view.zoomTo()
            view.setBackgroundColor('white')
            
            return view

        # Display the 3D structure
        view = get_structure_viz(atoms)
        html_str = view._make_html()
        st.components.v1.html(html_str, width=400, height=400)
        
        # Display structure information
        st.markdown("### Structure Information")
        atoms_info = {
            "Number of Atoms": len(atoms),
            "Chemical Formula": atoms.get_chemical_formula(),
            "Cell Dimensions": atoms.cell.cellpar() if atoms.cell else "No cell defined",
            "Atom Types": ", ".join(set(atoms.get_chemical_symbols()))
        }
        
        for key, value in atoms_info.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown('## Calculation Setup', unsafe_allow_html=True)
        
        # Display calculation details
        st.markdown("### Selected Model")
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Device:** {device}")
        
        st.markdown("### Selected Task")
        st.write(f"**Task:** {task}")
        
        if "Optimization" in task:
            st.write(f"**Max Steps:** {max_steps}")
            st.write(f"**Convergence Threshold:** {fmax} eV/Å")
            st.write(f"**Optimizer:** {optimizer}")
        
        # Run calculation button
        run_calculation = st.button("Run Calculation", type="primary")
        
        if run_calculation:
            try:
                with st.spinner("Running calculation..."):
                    # Copy atoms to avoid modifying the original
                    calc_atoms = atoms.copy()
                    
                    # Set up calculator based on selected model
                    if model_type == "MACE":
                        st.write("Setting up MACE calculator...")
                        calc = get_mace_model(model_path, device, selected_default_dtype)
                    else:  # FairChem
                        st.write("Setting up FairChem calculator...")
                        # Seems like the FairChem models use float32 and when switching from MACE 64 model to FairChem float32 model we get an error
                        # probably due to both sharing the underlying torch implementation
                        # So just a dummy statement to swithc torch to 32 bit
                        calc = get_mace_model('https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model', 'cpu', 'float32')
                        calc = get_fairchem_model(selected_model, model_path, device, selected_task_type)
                    # Attach calculator to atoms
                    calc_atoms.calc = calc
                    
                    # Perform the selected task
                    results = {}
                    
                    if task == "Energy Calculation":
                        # Calculate energy
                        energy = calc_atoms.get_potential_energy()
                        results["Energy"] = f"{energy:.6f} eV"
                    
                    elif task == "Energy + Forces Calculation":
                        # Calculate energy and forces
                        energy = calc_atoms.get_potential_energy()
                        forces = calc_atoms.get_forces()
                        max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
                        
                        results["Energy"] = f"{energy:.6f} eV"
                        results["Maximum Force"] = f"{max_force:.6f} eV/Å"
                    
                    elif task == "Geometry Optimization":
                        # Set up optimizer
                        if optimizer == "BFGS":
                            opt = BFGS(calc_atoms)
                        elif optimizer == "LBFGS":
                            opt = LBFGS(calc_atoms)
                        else:  # FIRE
                            opt = FIRE(calc_atoms)

                        # Streamlit placeholder for live-updating table
                        table_placeholder = st.empty()

                        # Container for log data
                        opt_log = []
                        # Attach the Streamlit logger to the optimizer
                        opt.attach(lambda: streamlit_log(opt), interval=1)
                        # Run optimization
                        st.write("Running geometry optimization...")
                        opt.run(fmax=fmax, steps=max_steps)
                        
                        # Get results
                        energy = calc_atoms.get_potential_energy()
                        forces = calc_atoms.get_forces()
                        max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
                        
                        results["Final Energy"] = f"{energy:.6f} eV"
                        results["Final Maximum Force"] = f"{max_force:.6f} eV/Å"
                        results["Steps Taken"] = opt.get_number_of_steps()
                        results["Converged"] = "Yes" if opt.converged() else "No"
                    
                    elif task == "Cell + Geometry Optimization":
                        # Set up optimizer with FrechetCellFilter
                        fcf = FrechetCellFilter(calc_atoms)
                        
                        if optimizer == "BFGS":
                            opt = BFGS(fcf)
                        elif optimizer == "LBFGS":
                            opt = LBFGS(fcf)
                        else:  # FIRE
                            opt = FIRE(fcf)
                            
                        # Streamlit placeholder for live-updating table
                        table_placeholder = st.empty()

                        # Container for log data
                        opt_log = []
                        # Attach the Streamlit logger to the optimizer
                        opt.attach(lambda: streamlit_log(opt), interval=1)
                        # Run optimization
                        st.write("Running cell + geometry optimization...")
                        opt.run(fmax=fmax, steps=max_steps)
                        
                        # Get results
                        energy = calc_atoms.get_potential_energy()
                        forces = calc_atoms.get_forces()
                        max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
                        
                        results["Final Energy"] = f"{energy:.6f} eV"
                        results["Final Maximum Force"] = f"{max_force:.6f} eV/Å"
                        results["Steps Taken"] = opt.get_number_of_steps()
                        results["Converged"] = "Yes" if opt.converged() else "No"
                        results["Final Cell Parameters"] = np.round(calc_atoms.cell.cellpar(), 4)
                
                    # Show results
                    st.success("Calculation completed successfully!")
                    st.markdown("### Results")
                    for key, value in results.items():
                        st.write(f"**{key}:** {value}")
                    
                    # If we did an optimization, show the final structure
                    if "Optimization" in task:
                        st.markdown("### Optimized Structure")
                        view = get_structure_viz(calc_atoms)
                        html_str = view._make_html()
                        st.components.v1.html(html_str, width=400, height=400)
                        
                        # Add download option for optimized structure
                        # First save the structure to a file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz") as tmp_file:
                            write(tmp_file.name, calc_atoms)
                            tmp_filepath = tmp_file.name
                        
                        # Read the content for downloading
                        with open(tmp_filepath, 'r') as file:
                            xyz_content = file.read()
                        
                        st.download_button(
                            label="Download Optimized Structure (XYZ)",
                            data=xyz_content,
                            file_name="optimized_structure.xyz",
                            mime="chemical/x-xyz"
                        )
                        
                        # Clean up the temp file
                        os.unlink(tmp_filepath)
                
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")
                st.error("Please make sure the structure is valid and compatible with the selected model.")
else:
    # Display instructions if no structure is loaded
    st.info("Please select a structure using the sidebar options to begin.")
    
    # Show some information about the app
    st.markdown("## About This App")
    st.write("""
    This app allows you to perform atomistic simulations using pre-trained foundational machine learning interatomic potentials (MLIPs) such as those from the MACE and FairChem libraries.
    
    ### Features:
    - Upload structure files (XYZ, CIF, POSCAR, etc.) or select from examples
    - Choose between MACE and FairChem ML models (more models coming soon)
    - Perform energy calculations, forces calculations, or geometry optimizations
    - Visualize structures in 3D
    - Download optimized structures
    
    ### Getting Started:
    1. Select an input method in the sidebar
    2. Choose a model and computational parameters
    3. Select a calculation task
    4. Run the calculation and analyze the results
    """)

# Footer
st.markdown("---")
st.markdown("Universal MLIP Playground App | Created with Streamlit, ASE, MACE, FairChem and ❤️")
st.markdown("Made by [Manas Sharma](https://manas.bragitoff.com/) in the groups of [Prof. Ananth Govind Rajan](https://www.agrgroup.org/) and [Prof. Sudeep Punnathanam](https://chemeng.iisc.ac.in/sudeep/)")
