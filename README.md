# Shoulder-angles-analysis
A 3D Slicer extension for computing shoulder radiographic metrics: beta angle, Critical Shoulder Angle (CSA), and Subacromial Space (SAS) using manually placed fiducial markups. 
The extension provides both numerical output (CSV) and visual overlays to verify measurement accuracy.

# Overview

"beta_angle_analysis" is designed for the quantitative evaluation of shoulder radiographs.
Users manually place anatomical fiducials on an X-ray, and the module computes:

  - Beta angle — angle between P1–P2 and P3–P4 (method described in article X).

  - Critical Shoulder Angle (CSA) — angle P4–P3–P5 (method described in article Y).

  - Subacromial Space (SAS) — perpendicular distance from P7 to the acromial line P5–P6.

Visual overlays (lines, angles, arcs) are automatically drawn to confirm that the calculations match anatomical expectations.

# Installation

## Installing 3D Slicer

To use Shoulder-angles-analysis with 3D Slicer, you'll need to download and install 3D Slicer. 
This extension supports the stable and preview versions of 3D Slicer, version 5.3 or higher. For more information on installing 3D Slicer, check out the 3D Slicer Documentation

## Add extension
Download or clone the repository.

In 3D Slicer, open Edit → Application Settings → Modules → Additional Module Paths:

<img width="917" height="667" alt="image" src="https://github.com/user-attachments/assets/97d970be-c633-44fb-af2b-8361d4df6609" />

Add the folder "Beta_angle_ana" containing the extension and restart Slicer. The path should look like, for example: "C:\Users\Name\Downloads\Shoulder-angles-analysis-main\Beta_angle_ana". 

The module will appear under Custom Modules as beta_angle_analysis. 
<img width="1920" height="821" alt="image" src="https://github.com/user-attachments/assets/c0155be5-5e4b-42b1-837c-b51f1ce7fb81" />


# Workflow

### Step 1 — Load the radiograph

Use the "add DICOM Data" module or drag-and-drop to add all the radiography you would like to analyse.

### Step 2 — Select the volume

Use the module’s input selector to choose the active X-ray image.

<img width="1349" height="606" alt="image" src="https://github.com/user-attachments/assets/6011af18-a4d3-4803-9f27-806b0dc33846" />

### Step 3 — Place fiducials

Click Place fiducials, then place exactly 7 points:

ID	Description
P1	First point on supraspinatus fossa cortical margin
P2	Second point on supraspinatus fossa cortical margin
P3	Glenoid lower
P4	Glenoid upper
P5	Lateral acromion
P6	Lower acromion
P7	Upper femoral head

The module automatically names points P1 → P7 in placement order.

### Step 4 — Compute metrics

Click Compute Parameters.

The module:

  - Detects the fiducials

  - Computes Beta angle, CSA, and SAS and draws the measurement overlays on the slice view. It saves the values into a CSV file

### Step 5 — Export fiducials (optional)

You can save an .fcsv file containing all landmark coordinates.

# Computed Measurements

1. Beta Angle

Angle between the lines formed by P1 → P2 (supraspinatus fossa border) and P3 → P4 (glenoid axis). The vertex is calculated as the intersection between both lines.
The method of computing is inspired by Maurer et al. 2012 study (doi.org/10.1016/j.jse.2011.07.010). 

2. Critical Shoulder Angle (CSA)

Angle at P3 (glenoid lower) formed by P4P3P5, according to Gomide et al. 2017 (doi.org/10.1016/j.rboe.2017.06.002.)

3. Subacromial Space (SAS)

Shortest (perpendicular) distance from: P7 (superior humeral head) to the line P5 → P6 (acromial margin)

The extension saves a file: "analysis_beta/<patientID>.csv" in the extension foler which contains:

  - beta_angle	in degree 
  - critical_shoulder_angle	in degree
  - subacromial_space	in mm


To ensure correctness, the module draws: the two lines forming the Beta angle, the intersection vertex and an arc depicting the Beta angle magnitude, the critical shoulder angle and the distance line of the SAS.
These are rendered as lightweight Markups (Curve or Line nodes) for clarity.

# Folder structure

beta_angle_analysis/

 ├── Beta_angle_ana.py    # main python script
 
 ├── Resources/
 
 │   ├── Icons/
 
 │   └── UI/
 
 ├── analysis_beta/        # Generated CSV files
 
 └── data/                 # Optional exported fiducials
             
