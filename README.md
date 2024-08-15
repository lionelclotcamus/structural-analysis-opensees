# Structural Analysis using OpenSees
This template app demonstrates how to perform a structural analysis of a 3D frame building and display the results using
Viktor. The structural analysis is conducted with OpenSees, a powerful and widely-used software for analyzing the 
structural response of structural and geotechnical systems to loads. By combining Viktor with OpenSees, structural 
analysis becomes easy to perform and accessible through a web browser.

Features include:
- Creation and visualization of the 3D frame building 
- Definition of loads 
- Running the structural analysis 
- Visualization of the deformed building

**Important** OpenSeesPy is free for research, education, and internal use. Commercial redistribution of OpenSeesPy, 
such as, but not limited to, an application or cloud-based service that uses import openseespy, requires a license 
similar to that required for commercial redistribution of OpenSees.exe. Contact 
[UC Berkeley](https://opensees.github.io/OpenSeesDocumentation/developer/license.html) for commercial licensing 
details.

### Step 1a: Create the 3D frame building
The width, length and number of floors can easily be adjusted. Additionally, the number of nodes per side can be 
changed.

![Step 1a](.viktor-template/step_1a.gif)

### Step 1b: Apply loads
For each load, a magnitude, direction and location can be defined.
- The magnitude is defined in kN.
- The direction can be defined in x, y and z-direction.
- The location can be selected with 'Select Geometry' and clicking on any node in the view on the right side.

Extra loads can be added by clicking on 'Add new row'. The load is visualized in the view on the right side with a red 
arrow.

![Step 1b](.viktor-template/step_1b.gif)

### Step 2: Running the analysis and viewing the results
After clicking on 'Next Step', the analysis can be performed by clicking on the 'Run analysis' button in the bottom 
right. Once the analysis is complete, both the undeformed and deformed building can be viewed on the right side. With a
color scale, the amount of deformation per element is also mapped onto the building. The deformation can be scaled with 
the 'Deformation scale factor'. If this is altered, the analysis must be performed again by clicking on 'Run analysis'.

![Step 2](.viktor-template/step_2.gif)