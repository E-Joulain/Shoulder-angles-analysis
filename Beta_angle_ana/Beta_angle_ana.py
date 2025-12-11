import logging
import os
import csv
from typing import Annotated

import vtk
import qt
import ctk
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode

try:
    import pandas as pd
except ImportError:
    # Install pandas if not available; this only runs when module is executed interactively
    slicer.util.infoDisplay("Installing pandas. Please wait...")
    with slicer.util.displayPythonShell():
        slicer.util.pip_install('pandas')
    import pandas as pd


class Beta_angle_ana(ScriptedLoadableModule):
    """
    Beta_angle_ana
    A scripted loadable module template adapted and refactored from your original script.

    This class registers the module in Slicer (title, categories, help text...) and
    connects the sample-data registration to application startup.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Beta_angle_ana")
        # Keep categories concise and correct
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Custome_model")]
        self.parent.dependencies = []
        self.parent.contributors = ["Elise Joulain (UNIGE)"]
        self.parent.helpText = _("""
This module assists with placing fiducials on shoulder X-rays and computing various
metrics (beta angle and related). It was refactored for clarity and stability.
""")
        self.parent.acknowledgementText = _(
            """
Developed at UNIGE. Based on ScriptedLoadableModule examples from 3D Slicer.
"""
        )

        # Register sample data after startup (optional)
        slicer.app.connect("startupCompleted()", registerSampleData)


def registerSampleData():
    """Register optional sample data shown in the Sample Data module."""
    try:
        import SampleData
    except Exception:
        return

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons")

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Beta_angle_ana",
        sampleName="Beta_angle_ana1",
        thumbnailFileName=os.path.join(iconsPath, "Beta_angle_ana1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Beta_angle_ana1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="Beta_angle_ana1",
    )

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Beta_angle_ana",
        sampleName="Beta_angle_ana2",
        thumbnailFileName=os.path.join(iconsPath, "Beta_angle_ana2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Beta_angle_ana2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="Beta_angle_ana2",
    )


@parameterNodeWrapper
class Beta_angle_anaParameterNode:
    """Parameter node wrapper that defines typed parameters and defaults."""

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


class Beta_angle_anaWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Refactored widget class: builds UI, connects callbacks and delegates heavy work to Logic."""

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.inputVolumeSelector = None
        self.markupNode = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        # Simple mapping of fiducial labels used in the UI table
        self.fiducialLabels = {
            "P1": "First point on cortical margin of the supraspinatus fossa",
            "P2": "Second point on cortical margin of the supraspinatus fossa",
            "P3": "Glenoid lower",
            "P4": "Glenoid upper",
            "P5": "Lateral acromion",
            "P6": "Lower acromion",
            "P7": "Upper femoral head"
        }

        # Main layout and form
        formLayout = qt.QFormLayout()
        self.layout.addLayout(formLayout)

        # Input volume selector
        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector.currentNodeChanged.connect(self.onInputVolumeChanged)
        self.layout.addWidget(qt.QLabel("Choose the volume on which you want to work:"))
        self.layout.addWidget(self.inputVolumeSelector)
        # Cosmetic style (optional)
        self.inputVolumeSelector.setStyleSheet("""
                    QComboBox {
                        background-color: #587880;
                        color: #587880;
                        font-weight: bold;
                        border: 1px solid #00796B;
                        padding: 3px;
                        border-radius: 4px;
                    }
                """)

        # Side selection
        self.selectionCollapsibleButton = ctk.ctkCollapsibleButton()
        self.selectionCollapsibleButton.text = "Enter the side of the inspected shoulder"
        self.layout.addWidget(self.selectionCollapsibleButton)
        self.selectionFormLayout = qt.QFormLayout(self.selectionCollapsibleButton)
        self.sideSelector = qt.QComboBox()
        self.sideSelector.addItem("Right")
        self.sideSelector.addItem("Left")
        self.selectionFormLayout.addRow("Select Side:", self.sideSelector)

        # Patient ID input
        self.patientIdLineEdit = qt.QLineEdit()
        self.patientIdLineEdit.setPlaceholderText("Enter patient ID")
        formLayout.addRow("Patient ID:", self.patientIdLineEdit)

        # Fiducial table
        self.fiducialTableCollapsible = ctk.ctkCollapsibleButton()
        self.fiducialTableCollapsible.text = "Here a table guiding the position of the needed fiducials nodes for the analysis"
        self.layout.addWidget(self.fiducialTableCollapsible)
        self.fiducialTableLayout = qt.QVBoxLayout(self.fiducialTableCollapsible)

        self.fiducialTable = qt.QTableWidget()
        self.fiducialTable.setRowCount(len(self.fiducialLabels))
        self.fiducialTable.setColumnCount(2)
        self.fiducialTable.setHorizontalHeaderLabels(["ID", "Description"])
        self.fiducialTable.horizontalHeader().setStretchLastSection(True)
        self.fiducialTable.verticalHeader().visible = False
        self.fiducialTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)

        for i, (fidID, label) in enumerate(self.fiducialLabels.items()):
            self.fiducialTable.setItem(i, 0, qt.QTableWidgetItem(fidID))
            self.fiducialTable.setItem(i, 1, qt.QTableWidgetItem(label))

        self.fiducialTableLayout.addWidget(self.fiducialTable)

        # Analysis section
        self.beta_angleGroup = ctk.ctkCollapsibleButton()
        self.beta_angleGroup.text = "Analysis of the image"
        self.layout.addWidget(self.beta_angleGroup)
        self.beta_angleLayout = qt.QVBoxLayout(self.beta_angleGroup)
        self.beta_angleGroup.setStyleSheet("background-color: #587880; color: white;")

        self.beta_anglePlaceButton = qt.QPushButton("Place fiducials nodes")
        self.beta_anglePlaceButton.connect('clicked(bool)', lambda: self.onPlaceFiducial("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_anglePlaceButton)

        self.beta_angleComputeButton = qt.QPushButton("Compute parameters")
        self.beta_angleComputeButton.connect('clicked(bool)', lambda: self.onCalculateMetrics("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_angleComputeButton)

        self.beta_angleExportButton = qt.QPushButton("Export fiducials nodes")
        self.beta_angleExportButton.connect('clicked(bool)', lambda: self.onExportFiducials("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_angleExportButton)

        # Reset button
        self.resetFiducialsButton = qt.QPushButton("Reset fiducials nodes")
        self.resetFiducialsButton.toolTip = "Remove all beta_angle placed fiducial nodes"
        self.resetFiducialsButton.connect('clicked(bool)', self.onResetFiducials_preop)
        self.beta_angleLayout.addWidget(self.resetFiducialsButton)

        # Layout margins and spacing
        self.beta_angleLayout.setContentsMargins(2, 2, 2, 2)
        self.beta_angleLayout.setSpacing(3)
        self.layout.setSpacing(3)
        self.layout.setContentsMargins(2, 2, 2, 2)

        # Initialize logic
        self.logic = Beta_angle_anaLogic()

    # ---------- Callbacks and helper methods ----------
    def onInputVolumeChanged(self, volumeNode):
        if volumeNode:
            slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)

    def onLoadDICOM(self):
        slicer.util.selectModule("DICOM")
        qt.QMessageBox.information(slicer.util.mainWindow(), "DICOM Import",
                                   "The DICOM browser has been opened.\nLoad your X-ray image, then return here to place fiducials.")

    def onResetFiducials_preop(self):
        scene = slicer.mrmlScene

        if not slicer.util.confirmYesNoDisplay("Are you sure you want to remove all fiducial nodes?"):
            return

        for node in list(scene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")):
            name = node.GetName() or ""
            if "beta_angle" in name.lower():
                scene.RemoveNode(node)

    def onPlaceFiducial(self, timepoint: str):
        """Activate placement of fiducials and create the markup node if needed."""
        nodeName = f"Fiducials_{timepoint}"
        self.markupNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not self.markupNode:
            self.markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", nodeName)
            # Register observer to automatically label points
            self.markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onFiducialAdded)

        if "beta_angle" in timepoint.lower():
            displayNode = self.markupNode.GetDisplayNode()
            if displayNode:
                displayNode.SetSelectedColor(0, 1, 0)  # green

        # Start place mode (1 point at a time)
        slicer.modules.markups.logic().StartPlaceMode(1)

    def onFiducialAdded(self, caller, event):
        """Automatically label new fiducials P1..PN up to the number of labels provided."""
        n = caller.GetNumberOfControlPoints()
        if n <= len(self.fiducialLabels):
            fidName = f"P{n}"
            caller.SetNthControlPointLabel(n - 1, fidName)
        else:
            # Stop placement if too many points
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
            slicer.util.showStatusMessage("Maximum fiducials placed — placement stopped.", 3000)

    def get_input_volume_spacing(self):
        """Return spacing (x, y) of selected input volume or None with an error message."""
        try:
            volumeNode = self.inputVolumeSelector.currentNode()
            if not volumeNode:
                slicer.util.errorDisplay("No volume selected in the input selector.")
                return None

            if not volumeNode.IsA("vtkMRMLScalarVolumeNode"):
                slicer.util.errorDisplay("Selected node is not a scalar volume.")
                return None

            spacing = volumeNode.GetSpacing()  # (x, y, z)
            if tuple(spacing) == (1.0, 1.0, 1.0):
                slicer.util.warningDisplay("Warning: spacing is default (1.0 mm). Image may lack DICOM geometry.")
            return spacing[0], spacing[1]

        except Exception as e:
            slicer.util.errorDisplay(f"Error retrieving pixel spacing: {str(e)}")
            return None


    def onExportFiducials(self, timepoint: str):
        nodeName = f"Fiducials_{timepoint}"
        node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not node:
            slicer.util.errorDisplay(f"No fiducials placed for {timepoint}")
            return

        moduleDir = os.path.dirname(os.path.abspath(__file__))
        dataDir = os.path.join(moduleDir, "data")
        os.makedirs(dataDir, exist_ok=True)

        patientID = self.patientIdLineEdit.text().strip()
        if not patientID:
            slicer.util.errorDisplay("Please enter a patient ID before exporting.")
            return

        fileName = "fiducials"
        fullFileName = f"{fileName}_{patientID}_{timepoint}.fcsv"
        filePath = os.path.join(dataDir, fullFileName)

        slicer.util.saveNode(node, filePath)
        slicer.util.showStatusMessage(f"Fiducials nodes saved to: {filePath}", 5000)

    def onCalculateMetrics(self, timepoint: str):
        """Compute all metrics from fiducials and draw them."""
        nodeName = f"Fiducials_{timepoint}"
        markupNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not markupNode:
            slicer.util.errorDisplay(f"No fiducials placed for {timepoint}")
            return

        patientID = self.patientIdLineEdit.text.strip()
        if not patientID:
            slicer.util.errorDisplay("Please enter a patient ID.")
            return

        points = self.logic.get_fiducial_positions(markupNode)
        if len(points) < len(self.fiducialLabels):
            slicer.util.errorDisplay(f"Not enough fiducials (required: {len(self.fiducialLabels)}).")
            return
        for label in points:
            points[label]['z'] = 0

        # --- Points ---
        P1 = np.array([points['P1']['x'], points['P1']['y'], points['P1']['z']])
        P2 = np.array([points['P2']['x'], points['P2']['y'], points['P2']['z']])
        P3 = np.array([points['P3']['x'], points['P3']['y'], points['P3']['z']])
        P4 = np.array([points['P4']['x'], points['P4']['y'], points['P4']['z']])
        P5 = np.array([points['P5']['x'], points['P5']['y'], points['P5']['z']])
        P6 = np.array([points['P6']['x'], points['P6']['y'], points['P6']['z']])
        P7 = np.array([points['P7']['x'], points['P7']['y'], points['P7']['z']])

        # --- Beta angle ---
        beta_angle = self.logic.compute_beta_angle(points)
            # Compute intersection in XY plane
        x1, y1 = P1[:2]
        x2, y2 = P2[:2]
        x3, y3 = P3[:2]
        x4, y4 = P4[:2]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            # Lines are parallel
            intersect = P2.copy()
        else:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            intersect = np.array([px, py, 0.0])

        # Draw arc with the intersection as vertex
        self.logic.draw_arc(intersect, P1, P4, beta_angle, name="BetaAngleArc", color=(1, 1, 0))

        # --- Critical Shoulder Angle (CSA) ---
        csa_angle = self.logic.compute_angle(P4, P3, P5)
        self.logic.draw_arc(P3, P4, P5, csa_angle, name="CSAArc", color=(0, 1, 0))

        # --- Subacromial space (SAS) ---
        sas_distance, closest_point = self.logic.distance_point_to_line(P7, P5, P6)
        self.logic.draw_line(P7, closest_point, name="SASDistance", color=(0, 0, 1))

        # Save metrics
        metrics = {
            "beta_angle": beta_angle,
            "critical_shoulder_angle": csa_angle,
            "subacromial_space": sas_distance
        }
        self.logic.saveMetricsToCSV(metrics, patientID)
        slicer.util.showStatusMessage(f"Beta: {beta_angle:.1f}°, CSA: {csa_angle:.1f}°, SAS: {sas_distance:.1f} mm",
                                      5000)


class Beta_angle_anaLogic(ScriptedLoadableModuleLogic):
    """Logic class for data processing, angle computation, and visualization."""

    def __init__(self):
        super().__init__()

    # --- Fiducial retrieval ---
    def get_fiducial_positions(self, markupNode):
        """Return a dict of fiducial positions keyed by label {P1: {x,y,z}, ...}."""
        points = {}
        if not markupNode:
            return points
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            pos = [0.0, 0.0, 0.0]
            markupNode.GetNthControlPointPosition(i, pos)
            points[label] = {"x": pos[0], "y": pos[1], "z": pos[2]}
        return points

    # --- Save data in a .csv file ---
    def saveMetricsToCSV(self, metrics, patientID):
        """
        Save metrics dictionary to CSV file in analysis_beta folder.
        """
        import os, csv
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        analysisDir = os.path.join(moduleDir, "analysis_beta")
        os.makedirs(analysisDir, exist_ok=True)
        fileName = f"{patientID}.csv"
        filePath = os.path.join(analysisDir, fileName)

        try:
            with open(filePath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])
                for key, value in metrics.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        value_str = ";".join(str(v) for v in np.array(value).flatten())
                    else:
                        value_str = str(value)
                    writer.writerow([key, value_str])
            slicer.util.showStatusMessage(f"Metrics saved to: {filePath}", 5000)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to save CSV: {e}")

    # --- Compute beta angle ---
    def compute_beta_angle(self, points):
        """Compute Beta angle (P1-P2 vs P3-P4)."""
        p1 = np.array([points['P1']['x'], points['P1']['y']])
        p2 = np.array([points['P2']['x'], points['P2']['y']])
        p3 = np.array([points['P3']['x'], points['P3']['y']])
        p4 = np.array([points['P4']['x'], points['P4']['y']])

        v1 = p2 - p1
        v2 = p4 - p3
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return 180 - np.degrees(np.arccos(dot))

    # --- Generic angle computation ---
    def compute_angle(self, A, B, C):
        """Compute angle at B formed by A-B-C."""
        BA = A - B
        BC = C - B
        BA_norm = BA / np.linalg.norm(BA)
        BC_norm = BC / np.linalg.norm(BC)
        dot = np.clip(np.dot(BA_norm, BC_norm), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    # --- Distance point-to-line ---
    def distance_point_to_line(self, P, A, B):
        """Distance from P to line A-B."""
        line_vec = B - A
        point_vec = P - A
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return 0, A
        proj = np.dot(point_vec, line_vec) / line_len
        closest = A + proj * line_vec / line_len
        dist = np.linalg.norm(P - closest)
        return dist, closest

    # --- Generate points along an arc ---
    def generate_arc_points(self, vertex, p1, p2, angle_deg, num_points=30):
        """
        Generate points along an arc centered at `vertex`, starting at `p1` and spanning `angle_deg` toward `p2`.
        """
        v1 = p1 - vertex
        radius = np.linalg.norm(v1)
        angle_rad = np.radians(angle_deg)

        points = []
        for t in np.linspace(0, 1, num_points):
            theta = t * angle_rad
            # 2D rotation in XY plane
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
            arc_point_2d = vertex[:2] + rot_matrix @ v1[:2]
            points.append([arc_point_2d[0], arc_point_2d[1], vertex[2]])
        return points

    # --- Draw arc ---
    def draw_arc(self, vertex, p1, p2, angle_deg, name="Arc", color=(1, 1, 0), thickness=0.001, num_points=30):
        """
        Draw an arc centered at `vertex`, starting at `p1` and spanning `angle_deg` toward `p2`.
        The arc is drawn in the XY plane.
        """
        # Remove old node with the same name
        for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsCurveNode"):
            if node.GetName() == name:
                slicer.mrmlScene.RemoveNode(node)

        # Generate points along the arc
        v_start = p1 - vertex
        v_end = p2 - vertex

        # Determine rotation direction (sign of angle)
        angle_rad = np.radians(angle_deg)
        points = []
        for t in np.linspace(0, 1, num_points):
            theta = t * angle_rad
            # 2D rotation in XY plane
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
            arc_point_xy = vertex[:2] + rot_matrix @ v_start[:2]
            points.append([arc_point_xy[0], arc_point_xy[1], vertex[2]])

        # Create Markups Curve Node
        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
        for pt in points:
            curveNode.AddControlPoint(pt)

        # Display properties
        displayNode = curveNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(*color)
            displayNode.SetLineThickness(thickness)
            displayNode.SetVisibility2D(True)
            displayNode.SetVisibility3D(True)

    # --- Draw line ---
    def draw_line(self, A, B, name="Line", color=(1, 0, 0), thickness=0.5):
        """Draw a line as a Markups Line Node."""
        # Remove previous line node with same name
        for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetName() == name:
                slicer.mrmlScene.RemoveNode(node)

        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        lineNode.AddControlPoint(A.tolist())
        lineNode.AddControlPoint(B.tolist())

        displayNode = lineNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(*color)
            displayNode.SetLineThickness(thickness)
            displayNode.SetVisibility2D(True)
            displayNode.SetVisibility3D(True)



class Beta_angle_anaTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Beta_angle_ana1()

    def test_Beta_angle_ana1(self):
        self.delayDisplay("Starting the test")
        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample("Beta_angle_ana1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        logic = Beta_angle_anaLogic()

        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
