# Overview


Miniaturizing transistors, the building blocks of integrated circuits, presents significant challenges for the semiconductor industry. Precisely measuring these features during production is crucial for high-quality chips. Existing in-line metrology techniques, such as optical critical-dimension (OCD) scatterometry and critical-dimension scanning electron microscopy (CD-SEM), are nearing their limits. OCD struggles with the inherent limitations of light and shrinking features, while CD-SEM, despite providing valuable insights, is restricted by sampling area and resolution. To overcome these obstacles, the industry is exploring X-ray-based metrology. X-rays, with their shorter wavelengths, allow for more precise analysis and are sensitive to variations in composition, providing richer data.


CD-SAXS (Critical Dimension Small Angle X-ray Scattering) is a promising technique for nanostructure electronics. It uses a transmission geometry, sending the beam through the sample and the 750 micrometer-thick silicon wafer. The x-ray spot size varies between 10-1000 μm, enabling the measurement of small patterned areas. Studies have shown CD-SAXS's effectiveness in characterizing the shape and spacing of nanometer-sized patterns.

This technique utilizes variable-angle transmission scattering. By rotating the sample, it can probe the vertical profile of the nanostructures, allowing for the reconstruction of their shape and composition in two or even three dimensions. This technique excels at reconstructing intricate shapes smaller than 15 nm and with spacing around 30 nm, dimensions crucial for the semiconductor industry.

Although several big companies are developing software for CD-SAXS, the technique is still in its infancy and there isn't a coherent package available. Thus this cdsaxs package is
aimed at providing simulation and fitting tools for CD-SAXS synchotron data. It can be separated into two main parts: simulations and fitter. Simulations generate synthetic data for a given model, while the fitter fits the model to the experimental data.  
