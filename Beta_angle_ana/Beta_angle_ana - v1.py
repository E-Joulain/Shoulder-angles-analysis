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
            "P1": "First omoplate",
            "P2": "Second omoplate",
            "P3": "up glaine",
            "P4": "down glaine",
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
                        background-color: #E8B952;
                        color: #00363A;
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
        self.fiducialTableCollapsible.text = "Here a table guiding the position of the needed fiducials nodes for the PHF analysis"
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
        self.beta_angleGroup.setStyleSheet("background-color: #50B550; color: white;")

        self.beta_anglePlaceButton = qt.QPushButton("Place fiducials nodes")
        self.beta_anglePlaceButton.connect('clicked(bool)', lambda: self.onPlaceFiducial("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_anglePlaceButton)

        self.beta_angleComputeButton = qt.QPushButton("Compute Parameters (beta_angle)")
        self.beta_angleComputeButton.connect('clicked(bool)', lambda: self.onCalculateMetrics("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_angleComputeButton)

        self.beta_angleExportButton = qt.QPushButton("Export Fiducials (beta_angle)")
        self.beta_angleExportButton.connect('clicked(bool)', lambda: self.onExportFiducials("beta_angle"))
        self.beta_angleLayout.addWidget(self.beta_angleExportButton)

        # Reset button
        self.resetFiducialsButton = qt.QPushButton("Reset Fiducials (beta_angle)")
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
        nodeName = f"Fiducials_{timepoint}"
        markupNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not markupNode:
            slicer.util.errorDisplay(f"No fiducials placed for {timepoint}")
            return

        # Patient ID
        patientID = self.patientIdLineEdit.text.strip()
        if not patientID:
            slicer.util.errorDisplay("Please enter a patient ID.")
            return

        # Save to temp and use logic to read coordinates
        fcsv_path = os.path.join(slicer.app.temporaryPath, f"temp_{timepoint}.fcsv")
        slicer.util.saveNode(markupNode, fcsv_path)
        points = self.logic.loadFCSV(fcsv_path)

        if len(points) < len(self.fiducialLabels):
            slicer.util.errorDisplay(
                f"Not enough fiducials for {timepoint} (required: {len(self.fiducialLabels)})."
            )
            return

        # Convert list → dict indexed by label
        point_dict = {p["label"]: p for p in points}

        # Compute beta angle
        beta_angle = self.logic.compute_beta_angle(point_dict)
        print(f"beta_angle: {beta_angle}")

        self.logic.draw_beta_angle_lines(point_dict)

        # After computing beta_angle
        metrics = {"beta_angle": beta_angle}
        self.logic.saveMetricsToCSV(metrics, patientID)

    #   slicer.util.showStatusMessage("Metrics computed and saved.", 5000)

class Beta_angle_anaLogic(ScriptedLoadableModuleLogic):
    """Logic class that contains data processing and I/O helpers."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getFiducialPoints(self, markupNode):
        coords = []
        for i in range(markupNode.GetNumberOfFiducials()):
            pos = [0.0, 0.0, 0.0]
            markupNode.GetNthFiducialPosition(i, pos)
            coords.append((pos[0], pos[1]))
        return np.array(coords)

    def loadFCSV(self, filePath: str):
        """Simple .fcsv loader that returns list of dicts with x,y,z,label."""
        points = []
        try:
            with open(filePath, 'r') as f:
                header = None
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split(',')
                    # Typical Slicer .fcsv has format: index,x,y,z,ow,ox,oy,oz,vis,sel,lock,label
                    if len(parts) < 12:
                        continue
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        label = parts[11]
                    except ValueError:
                        continue
                    points.append({'label': label, 'x': x, 'y': y, 'z': z})
        except Exception as e:
            logging.error(f"Failed to read FCSV {filePath}: {e}")
        return points

    def compute_beta_angle(self, points):
        """"
        points : dictionnaire {label, x, y, z}
        P1,P2 : omoplate
        P3,P4 : glène
        """

        # Retrieve required points
        p1 = np.array([points['P1']['x'], points['P1']['y']])
        p2 = np.array([points['P2']['x'], points['P2']['y']])
        p3 = np.array([points['P3']['x'], points['P3']['y']])
        p4 = np.array([points['P4']['x'], points['P4']['y']])

        # Build vectors
        v1 = p2 - p1      # Scapular axis
        v2 = p4 - p3      # Glenoid axis

        # Normalize
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Compute angle in degrees
        dot = np.dot(v1_norm, v2_norm)
        dot = np.clip(dot, -1.0, 1.0)  # avoid numerical errors

        angle_deg = np.degrees(np.arccos(dot))

        return angle_deg

    def getParameterNode(self):
        return Beta_angle_anaParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """Example processing wrapper that calls the ThresholdScalarVolume CLI module."""
        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info("Processing started")

        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams,
                                 wait_for_completion=True, update_display=showResult)
        try:
            slicer.mrmlScene.RemoveNode(cliNode)
        except Exception:
            pass

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def saveMetricsToCSV(self, metrics, patientID):
        """
        Save metrics dictionary to CSV file in analysis_beta folder.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics to save. e.g., {"beta_angle": 903480}
        patientID : str
            Patient ID to use as filename.
        """
        # Create analysis_beta folder in the module directory
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        analysisDir = os.path.join(moduleDir, "analysis_beta")
        os.makedirs(analysisDir, exist_ok=True)

        # Construct file path using patient ID
        fileName = f"{patientID}.csv"
        filePath = os.path.join(analysisDir, fileName)

        try:
            with open(filePath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])  # Header
                for key, value in metrics.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        value_str = ";".join(str(v) for v in np.array(value).flatten())
                    else:
                        value_str = str(value)
                    writer.writerow([key, value_str])
            slicer.util.showStatusMessage(f"Metrics saved to: {filePath}", 5000)
            logging.info(f"Metrics saved to: {filePath}")
            print(f"file saved to:{filePath}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to save CSV: {e}")
            logging.error(f"Failed to save CSV {filePath}: {e}")

    def load_csv_data(self, metric, id_op):
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        dataDir = os.path.join(moduleDir, "data")
        FileName = f"{metric}_{id_op}.csv"
        filePath = os.path.join(dataDir, FileName)
        if not os.path.exists(filePath):
            raise FileNotFoundError(filePath)
        data = pd.read_csv(filePath)
        return data

    def draw_beta_angle_lines(self, points):
        """
        Draw lines connecting P1-P2 (scapular axis) and P3-P4 (glenoid axis)
        """
        # Remove previous lines
        for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode"):
            slicer.mrmlScene.RemoveNode(node)

        print("Here")
        # Scapular axis
        line1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "ScapularAxis")
        line1.AddControlPoint([points['P1']['x'], points['P1']['y'], 0])
        print(f"{points['P1']['x']}, {points['P1']['y']}, {points['P1']['z']}")
        line1.AddControlPoint([points['P2']['x'], points['P2']['y'], 0])
        line1.GetDisplayNode().SetSelectedColor(1, 0, 0)  # Red

        # Glenoid axis
        line2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "GlenoidAxis")
        line2.AddControlPoint([points['P3']['x'], points['P3']['y'], points['P3']['z']])
        line2.AddControlPoint([points['P4']['x'], points['P4']['y'], points['P4']['z']])
        line2.GetDisplayNode().SetSelectedColor(0, 0, 1)




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
