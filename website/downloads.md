---
layout: page
title: "Downloads & Resources"
description: "Download the framework, access documentation, and get additional resources"
---

## Software Downloads

### Latest Release: Version 1.0.0

<div class="download-section">
    <div class="download-card primary">
        <div class="download-icon">📦</div>
        <h3>Complete Framework</h3>
        <p>Full adaptive multi-fidelity optimization framework with all components</p>
        <div class="download-stats">
            <span>Size: 45 MB</span> • <span>Updated: Aug 2024</span>
        </div>
        <div class="download-buttons">
            <a href="https://github.com/aerospace-optimization/amf-sbo/archive/v1.0.0.zip" class="btn btn-primary">
                Download ZIP
            </a>
            <a href="https://github.com/aerospace-optimization/amf-sbo" class="btn btn-outline">
                View on GitHub
            </a>
        </div>
    </div>
</div>

### Installation Options

#### Option 1: Git Clone (Recommended)
```bash
git clone https://github.com/aerospace-optimization/amf-sbo.git
cd amf-sbo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

#### Option 2: Direct Download
1. Download the ZIP file from the link above
2. Extract to your desired directory
3. Follow the installation instructions in README.md

#### Option 3: PyPI Installation (Coming Soon)
```bash
pip install amf-sbo
```

### System Requirements

#### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM
- **Storage**: 1 GB available space
- **Processor**: Dual-core CPU

#### Recommended Requirements
- **Memory**: 8 GB RAM or higher
- **Processor**: Quad-core CPU or higher
- **Storage**: 5 GB available space (for examples and results)
- **Graphics**: Dedicated GPU (optional, for visualization acceleration)

## Documentation Downloads

### Complete Documentation Package

<div class="doc-downloads">
    <div class="doc-download-item">
        <div class="doc-icon">📘</div>
        <div class="doc-info">
            <h4>User Guide</h4>
            <p>Comprehensive tutorials and step-by-step examples</p>
        </div>
        <div class="doc-actions">
            <a href="/docs/USER_GUIDE.pdf" class="btn-small">PDF</a>
            <a href="/docs/USER_GUIDE.html" class="btn-small">HTML</a>
        </div>
    </div>
    
    <div class="doc-download-item">
        <div class="doc-icon">🔧</div>
        <div class="doc-info">
            <h4>API Reference</h4>
            <p>Complete API documentation with all classes and methods</p>
        </div>
        <div class="doc-actions">
            <a href="/docs/API_REFERENCE.pdf" class="btn-small">PDF</a>
            <a href="/docs/API_REFERENCE.html" class="btn-small">HTML</a>
        </div>
    </div>
    
    <div class="doc-download-item">
        <div class="doc-icon">📊</div>
        <div class="doc-info">
            <h4>Technical Methodology</h4>
            <p>Mathematical formulations and implementation details</p>
        </div>
        <div class="doc-actions">
            <a href="/docs/TECHNICAL_METHODOLOGY.pdf" class="btn-small">PDF</a>
            <a href="/docs/TECHNICAL_METHODOLOGY.html" class="btn-small">HTML</a>
        </div>
    </div>
    
    <div class="doc-download-item">
        <div class="doc-icon">🤝</div>
        <div class="doc-info">
            <h4>Contributing Guide</h4>
            <p>Guidelines for contributing to the project</p>
        </div>
        <div class="doc-actions">
            <a href="/docs/CONTRIBUTING.pdf" class="btn-small">PDF</a>
            <a href="/docs/CONTRIBUTING.html" class="btn-small">HTML</a>
        </div>
    </div>
</div>

### Quick Reference Cards

- **[Algorithm Comparison Chart](resources/algorithm_comparison.pdf)** - Quick reference for choosing optimization algorithms
- **[Parameter Bounds Guide](resources/parameter_bounds.pdf)** - Recommended parameter ranges for different aircraft and spacecraft types
- **[Fidelity Strategy Guide](resources/fidelity_strategies.pdf)** - When to use different fidelity switching strategies
- **[Troubleshooting Checklist](resources/troubleshooting.pdf)** - Common issues and solutions

## Example Files and Data

### Complete Example Suite

<div class="example-downloads">
    <div class="example-category">
        <h4>🛩️ Aircraft Examples</h4>
        <ul>
            <li><a href="examples/aircraft_commercial_optimization.py">Commercial Airliner Optimization</a></li>
            <li><a href="examples/aircraft_regional_design.py">Regional Aircraft Design</a></li>
            <li><a href="examples/aircraft_business_jet.py">Business Jet Configuration</a></li>
            <li><a href="examples/aircraft_multi_objective.py">Multi-Objective Aircraft Design</a></li>
        </ul>
    </div>
    
    <div class="example-category">
        <h4>🛰️ Spacecraft Examples</h4>
        <ul>
            <li><a href="examples/spacecraft_earth_observation.py">Earth Observation Satellite</a></li>
            <li><a href="examples/spacecraft_communication.py">Communication Satellite</a></li>
            <li><a href="examples/spacecraft_interplanetary.py">Interplanetary Probe</a></li>
            <li><a href="examples/spacecraft_constellation.py">Satellite Constellation</a></li>
        </ul>
    </div>
    
    <div class="example-category">
        <h4>🔬 Advanced Examples</h4>
        <ul>
            <li><a href="examples/robust_optimization.py">Robust Design Under Uncertainty</a></li>
            <li><a href="examples/sensitivity_analysis.py">Global Sensitivity Analysis</a></li>
            <li><a href="examples/fidelity_comparison.py">Fidelity Strategy Comparison</a></li>
            <li><a href="examples/parallel_optimization.py">Parallel Processing Example</a></li>
        </ul>
    </div>
</div>

### Example Data Files

- **[Aircraft Performance Database](data/aircraft_performance_data.csv)** - Comprehensive aircraft design parameters and performance metrics
- **[Spacecraft Configuration Database](data/spacecraft_orbital_data.csv)** - Satellite and spacecraft design configurations
- **[Environmental Conditions](data/environmental_uncertainty_data.csv)** - Atmospheric and space environment data
- **[Optimization Results](data/optimization_convergence_data.csv)** - Sample optimization convergence data

### Jupyter Notebooks

Interactive tutorials and examples:

- **[Quick Start Notebook](notebooks/01_Quick_Start.ipynb)** - Basic framework introduction
- **[Aircraft Optimization Tutorial](notebooks/02_Aircraft_Optimization.ipynb)** - Step-by-step aircraft design
- **[Spacecraft Design Notebook](notebooks/03_Spacecraft_Design.ipynb)** - Complete spacecraft optimization
- **[Multi-Objective Analysis](notebooks/04_Multi_Objective.ipynb)** - Pareto front analysis
- **[Uncertainty Quantification](notebooks/05_Robust_Design.ipynb)** - Robust design methods
- **[Advanced Techniques](notebooks/06_Advanced_Features.ipynb)** - Custom implementations

## Research Publications

### Primary Publications

1. **"Adaptive Multi-Fidelity Optimization for Aerospace Design: A Comprehensive Framework"**  
   *Journal of Computational Aerospace Engineering* (2024)  
   [Download PDF](publications/adaptive_multifidelity_2024.pdf) | [View Online](https://doi.org/10.xxxx/xxxx)

2. **"Intelligent Fidelity Switching Strategies in Multi-Fidelity Optimization"**  
   *Optimization and Engineering* (2024)  
   [Download PDF](publications/fidelity_switching_2024.pdf) | [View Online](https://doi.org/10.xxxx/xxxx)

3. **"Robust Aerospace Design Under Uncertainty: A Multi-Fidelity Approach"**  
   *AIAA Journal* (2024)  
   [Download PDF](publications/robust_design_2024.pdf) | [View Online](https://doi.org/10.xxxx/xxxx)

### Conference Presentations

- **AIAA SciTech 2024**: "Adaptive Fidelity Management in Aerospace Optimization"
- **WCCM 2024**: "Multi-Fidelity Methods for Large-Scale Engineering Optimization"
- **AIAA Aviation 2024**: "Uncertainty Quantification in Aircraft Design Optimization"

## Datasets and Benchmarks

### Validation Datasets

<div class="dataset-grid">
    <div class="dataset-card">
        <h4>🛩️ Aircraft Validation Set</h4>
        <p>30 validated aircraft configurations with experimental data</p>
        <div class="dataset-stats">
            <span>Size: 2.1 MB</span> • <span>Format: CSV</span>
        </div>
        <a href="datasets/aircraft_validation_dataset.csv" class="btn-small">Download</a>
    </div>
    
    <div class="dataset-card">
        <h4>🛰️ Spacecraft Database</h4>
        <p>40 spacecraft configurations with mission parameters</p>
        <div class="dataset-stats">
            <span>Size: 1.8 MB</span> • <span>Format: CSV</span>
        </div>
        <a href="datasets/spacecraft_database.csv" class="btn-small">Download</a>
    </div>
    
    <div class="dataset-card">
        <h4>🌍 Environmental Data</h4>
        <p>Atmospheric and space environment conditions</p>
        <div class="dataset-stats">
            <span>Size: 950 KB</span> • <span>Format: CSV</span>
        </div>
        <a href="datasets/environmental_conditions.csv" class="btn-small">Download</a>
    </div>
</div>

### Benchmark Problems

Standard optimization test problems for algorithm comparison:

- **[Aerospace Benchmark Suite](benchmarks/aerospace_benchmark_suite.zip)** - 15 aerospace design optimization problems
- **[Multi-Objective Test Problems](benchmarks/multi_objective_problems.zip)** - 8 multi-objective aerospace problems
- **[Uncertainty Benchmark Set](benchmarks/uncertainty_problems.zip)** - Problems with defined uncertainties

## Software Tools and Utilities

### Standalone Tools

- **[Parameter Bounds Calculator](tools/bounds_calculator.exe)** - GUI tool for determining parameter bounds
- **[Fidelity Cost Estimator](tools/fidelity_estimator.exe)** - Estimate computational costs for different fidelity levels
- **[Result Analyzer](tools/result_analyzer.exe)** - Standalone tool for analyzing optimization results

### Configuration Templates

- **[Aircraft Configuration Template](configs/aircraft_template.json)** - Pre-configured aircraft optimization setup
- **[Spacecraft Configuration Template](configs/spacecraft_template.json)** - Pre-configured spacecraft optimization setup
- **[Multi-Objective Template](configs/multi_objective_template.json)** - Multi-objective optimization configuration

### Visualization Resources

- **[Professional Color Schemes](resources/color_schemes.json)** - Aerospace industry color palettes
- **[Plot Templates](resources/plot_templates.json)** - Pre-configured plot styles
- **[Report Templates](resources/report_templates/)** - LaTeX and Word templates for technical reports

## Development Resources

### Source Code

- **[GitHub Repository](https://github.com/aerospace-optimization/amf-sbo)** - Complete source code with version history
- **[Development Branch](https://github.com/aerospace-optimization/amf-sbo/tree/develop)** - Latest development features
- **[Issue Tracker](https://github.com/aerospace-optimization/amf-sbo/issues)** - Bug reports and feature requests

### Development Tools

- **[Development Setup Script](scripts/setup_development.sh)** - Automated development environment setup
- **[Testing Framework](scripts/run_tests.sh)** - Comprehensive test suite runner
- **[Documentation Builder](scripts/build_docs.sh)** - Generate documentation locally

## License and Legal

### Software License

This project is released under the **MIT License**, which allows:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution  
- ✅ Private use

**Requirements:**
- Include license and copyright notice
- No warranty or liability

[View Full License](LICENSE.txt)

### Citation

If you use this software in your research, please cite:

```bibtex
@software{amf_sbo_2024,
  title={Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems},
  author={AMF-SBO Development Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/aerospace-optimization/amf-sbo}
}
```

## Support and Community

### Getting Help

- **[Documentation]({{ '/documentation/' | relative_url }})** - Comprehensive guides and tutorials
- **[GitHub Discussions](https://github.com/aerospace-optimization/amf-sbo/discussions)** - Community Q&A
- **[Issue Tracker](https://github.com/aerospace-optimization/amf-sbo/issues)** - Bug reports and feature requests
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/amf-sbo)** - Technical questions

### Community Resources

- **[User Forum](https://forum.amf-sbo.com)** - Community discussions and tips
- **[Slack Workspace](https://amf-sbo.slack.com)** - Real-time chat and collaboration
- **[Monthly Webinars](https://webinars.amf-sbo.com)** - Live demonstrations and Q&A sessions
- **[Newsletter](https://newsletter.amf-sbo.com)** - Updates and announcements

### Contributing

We welcome contributions! See our **[Contributing Guide]({{ '/docs/CONTRIBUTING.html' | relative_url }})** for:

- How to report bugs
- How to suggest features
- How to contribute code
- Code style guidelines
- Development setup

## Contact Information

### Research Team

- **Primary Contact**: research@amf-sbo.com
- **Technical Support**: support@amf-sbo.com
- **Collaboration Inquiries**: partnerships@amf-sbo.com

### Academic Collaborations

We actively seek collaborations with:

- Aerospace engineering departments
- Optimization research groups
- Industry R&D organizations
- Government research laboratories

For collaboration inquiries, please contact: collaborations@amf-sbo.com

---

*All downloads are provided as-is under the MIT License. See individual files for specific terms and conditions.*