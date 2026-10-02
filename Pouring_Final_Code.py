#! /usr/bin/env python3

import sys
import os
import time
import threading
import re
from collections import deque

import serial
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================================
# KINOVA IMPORTS
# ==========================================================

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# This should be the same utilities.py you already use
import utilities


# ==========================================================
# 1. SETTINGS
# ==========================================================

CAMERA_INDEX = 1

model = YOLO("yolo11n.pt")

CONFIDENCE_THRESHOLD = 0.35

WORKSPACE_MARGIN = 5


# ==========================================================
# SAFE ROBOT POSE
# ==========================================================

SAFE_Z = 0.320

# Pickup height measured manually with fingers around
# the middle of the cup.
PICK_Z = 0.060


# Direct gripper position commands.
GRIPPER_OPEN = 0.00

# Close value previously used for this cup/gripper setup.
# If needed later, this can be fine-tuned slightly.
GRIPPER_CLOSE = 0.80

# ==========================================================
# DESTINATION CUP DETECTION
# ==========================================================

# Cup 2 must be at least this far from the original
# position of Cup 1.
DEST_MIN_DISTANCE = 0.12   # metres = 12 cm

DEST_WINDOW_SIZE = 15
DEST_MIN_DETECTIONS = 8

# ==========================================================
# DESTINATION CUP / SCALE ALIGNMENT OFFSET-----------------------------!!!!!!!!!---------------CHANGE
# ==========================================================

# Final repeated destination Y calibration.
DESTINATION_Y_OFFSET = -0.235   # metres = -235 mm
# Final repeated destination X calibration.
DESTINATION_X_OFFSET = -0.015    # metres = -15 mm

# ==========================================================
# STEP 4 - CALIBRATED ACTUATOR 7 POUR
# ==========================================================

# Experimentally measured with water:
# - Water starts pouring at approximately 114.60 degrees.
# - Cup is fully emptied / maximum pour at approximately 230.00 degrees.
#
# IMPORTANT:
# Actuators 1-6 are kept at their CURRENT measured positions.
# Only Actuator 7 is changed during the pour.
POUR_START_ACTUATOR_7 = 114.60
POUR_FULL_ACTUATOR_7 = 230.00

# Short pauses used during the automatic pour.
POUR_START_PAUSE = 0.5
POUR_FULL_HOLD = 1.5

# ==========================================================
# STEP 5 - KERN PCB6000-0 SCALE INTEGRATION
# ==========================================================

# Settings taken from the working target-weight scale test.
SCALE_PORT = "COM10"
SCALE_BAUDRATE = 9600
SCALE_TIMEOUT = 1

# Let the receiving cup / scale settle after the robot
# returns upright before taking the final reading.
FINAL_WEIGHT_SETTLE_TIME = 2.0

# ==========================================================
# VOLUME-BASED POURING - EXPERIMENTAL LIQUID DENSITIES
# ==========================================================
#
# Densities obtained from the laboratory graduated-cylinder
# measurements. The robot accepts a desired volume in mL,
# converts it to target mass using:
#
#     target_mass_g = density_g_per_ml * target_volume_ml
#
# The existing KERN scale controller then works in grams.
LIQUID_DENSITIES = {
    "water": 1.000,
    "mustard_oil": 0.900,
    "dishwashing_liquid": 1.019,
    "honey": 1.471,
}

LIQUID_DISPLAY_NAMES = {
    "water": "Water",
    "mustard_oil": "Mustard Oil",
    "dishwashing_liquid": "Dishwashing Liquid",
    "honey": "Honey",
}

# ==========================================================
# STEP 6B - PULSE-AND-WEIGH TARGET CONTROL
# ==========================================================

# Previous continuous-ramp test:
#   target = 25 g
#   final  = 50 g
#
# The large overshoot shows that scale feedback + falling
# liquid lag is too large while the cup stays continuously
# tilted.
#
# New strategy:
#   1. Tilt briefly.
#   2. Return fully upright to stop liquid flow.
#   3. Wait for the KERN scale to settle.
#   4. Read the added liquid mass.
#   5. Repeat only if more liquid is required.

# Initial pulse angle. Water starts flowing near 114.60 deg,
# so this begins only a few degrees above that point.
PULSE_START_ANGLE = 120.0

# Adaptive pulse-angle limits.
#
# Previous testing showed that no liquid was flowing by 135 deg.
# Allow the controller to keep searching gradually up to 175 deg.
PULSE_MIN_ANGLE = 116.5
PULSE_MAX_ANGLE = 175.0

# How much the pulse angle may change between cycles.
PULSE_ANGLE_UP_STEP = 4.0
PULSE_ANGLE_DOWN_STEP = 2.0

# ==========================================================
# FINAL 3 g FINE-ANGLE CONTROL - ALL LIQUIDS
# ==========================================================
# When 3 g or less remains:
#   1. Back off the current pour angle by 4 degrees ONCE.
#   2. Then use smaller +1.5 degree increases whenever
#      more angle is required.
# Applies to all liquids.
FINAL_FINE_REMAINING_G = 10.0
FINAL_FINE_ANGLE_BACKOFF = 4.0
FINAL_FINE_ANGLE_UP_STEP = 1.5

# Short pulse durations.
# Near the target, reduce TIME rather than forcing the angle lower.
#
# WATER and MUSTARD OIL:
# Keep the already validated pulse timing unchanged.
COARSE_PULSE_HOLD = 0.20
FINE_PULSE_HOLD = 0.08

# VISCOUS LIQUID HOLD TIMES
#
# Dishwashing liquid and honey flow more slowly than water/oil.
# These longer holds give the liquid time to leave the source
# cup before Actuator 7 returns upright.
#
# Only the HOLD TIME changes for these liquids.
# Adaptive angles, scale feedback and all other controller
# settings remain unchanged.
DISHWASH_COARSE_PULSE_HOLD = 0.45
DISHWASH_FINE_PULSE_HOLD = 0.18

HONEY_COARSE_PULSE_HOLD = 0.80
HONEY_FINE_PULSE_HOLD = 0.30

# ==========================================================
# VISCOUS-LIQUID FLOW-HOLD CONTROL
# ==========================================================
#
# Dishwashing liquid and honey are handled differently from
# water and mustard oil.
#
# After reaching the selected Actuator-7 angle, the robot
# STAYS tilted and checks the KERN scale repeatedly.
#
# If measured liquid mass keeps increasing, the robot keeps
# holding the SAME angle.
#
# When there is no measurable increase for several consecutive
# checks, the robot returns upright, increases the angle by
# PULSE_ANGLE_UP_STEP, and tries again.
#
# A maximum hold time is included as a safety limit so the
# robot cannot remain tilted indefinitely.

VISCOUS_FLOW_GAIN_THRESHOLD_G = 0.5
VISCOUS_NO_FLOW_CONFIRMATIONS = 3

DISHWASH_MIN_HOLD_TIME = 1.50
DISHWASH_FLOW_CHECK_INTERVAL = 0.75
DISHWASH_MAX_HOLD_TIME = 8.0

HONEY_MIN_HOLD_TIME = 2.50
HONEY_FLOW_CHECK_INTERVAL = 0.60
HONEY_MAX_HOLD_TIME = 8.0

# Switch to fine control when this much liquid remains.
FINE_REMAINING_G = 7.0

# Once very close to target, stop slightly early to allow for
# any final drops that arrive after the cup returns upright.
EARLY_STOP_MARGIN_G = 1.0

# Wait after every pulse with the cup upright before reading
# the KERN scale.
PULSE_SETTLE_TIME = 1.20

# If a pulse adds almost nothing, slightly increase the next
# pulse angle. If a pulse adds a lot, reduce the next angle.
MIN_USEFUL_PULSE_GAIN_G = 0.5
LARGE_PULSE_GAIN_G = 5.0

# Maximum number of dispense/weigh cycles.
MAX_POUR_PULSES = 60

# Maximum time allowed to receive a fresh scale reading.
SCALE_READ_TIMEOUT = 5.0

# Validated movement limits.
# The camera can still DETECT a cup outside these limits,
# but the robot will NOT be allowed to move there.
MAX_SAFE_X = 0.800
MAX_SAFE_REACH = 0.805

THETA_Y = -1.1
THETA_Z = 92.7


# ==========================================================
# JSON CUP PICK / LIFT ORIENTATION
# Taken from:
# "#2:CUP PICK and DOWN"
#
# IMPORTANT:
# Only the ORIENTATION is reused from the old JSON action.
# The old JSON X/Y/Z values are NOT used because the robot,
# camera and table setup have changed.
# ==========================================================

PICK_THETA_X = 88.608345
PICK_THETA_Y = -1.817637
PICK_THETA_Z = 94.479439

# ==========================================================
# JSON PICK ALIGNMENT CORRECTION
# ==========================================================
# Measured after rotating into the JSON pick orientation:
#
#   commanded X       = 0.608 m
#   manually centred X = 0.578 m
#
# Therefore:
# Current JSON pickup calibration is defined below.
#
# IMPORTANT:
# This offset is ONLY used for the JSON pickup pose.
# It does not alter camera calibration, normal hover centring,
# Cup-2 destination alignment, or pouring calibration.
JSON_PICK_X_OFFSET = 0.000    # - = back + = forwrad
JSON_PICK_Y_OFFSET = 0.000   # - = Left + = Right

# ==========================================================
# KERN SCALE HELPERS
# ==========================================================

def connect_kern_scale():
    """
    Connect to the KERN PCB6000-0 using the same serial
    settings that worked in the standalone scale test.
    """

    print()
    print("======================================")
    print("CONNECTING TO KERN SCALE")
    print("======================================")

    try:

        ser = serial.Serial(
            port=SCALE_PORT,
            baudrate=SCALE_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=SCALE_TIMEOUT
        )

        print(
            f"KERN scale connected on {SCALE_PORT}."
        )

        return ser

    except serial.SerialException as exc:

        print()
        print(
            f"ERROR: Could not open KERN scale on "
            f"{SCALE_PORT}."
        )
        print(exc)

        return None


def read_kern_weight(ser):
    """
    Wait for one valid numeric weight reading from the scale.
    Uses the same numeric parsing as the working standalone
    scale program.
    """

    while True:

        data = ser.readline()

        if not data:
            continue

        text_line = (
            data.decode(
                "ascii",
                errors="ignore"
            )
            .strip()
        )

        match = re.search(
            r"[-+]?\d+(?:\.\d+)?",
            text_line
        )

        if not match:
            continue

        return float(
            match.group()
        )


def capture_receiving_cup_tare(ser):
    """
    Record the starting weight of Cup 2 before any liquid
    is poured into it.
    """

    print()
    print("======================================")
    print("SET RECEIVING CUP STARTING WEIGHT")
    print("======================================")
    print(
        "Cup 2 must be sitting on the KERN scale."
    )
    print(
        "Make sure the scale reading is stable."
    )
    print(
        "If your scale sends readings using PRINT, "
        "press PRINT after continuing."
    )

    input(
        "Press ENTER when Cup 2 is stable on the scale..."
    )

    ser.reset_input_buffer()
    time.sleep(0.5)

    print(
        "Waiting for Cup 2 starting weight..."
    )

    tare_weight = read_kern_weight(
        ser
    )

    print()
    print(
        f"Cup 2 starting weight = "
        f"{tare_weight:.1f} g"
    )

    return tare_weight


def read_final_liquid_weight(
    ser,
    tare_weight
):
    """
    After pouring, read the total scale value and subtract
    the stored Cup 2 starting weight.
    """

    print()
    print(
        f"Waiting {FINAL_WEIGHT_SETTLE_TIME:.1f} s "
        "for the scale to settle..."
    )

    time.sleep(
        FINAL_WEIGHT_SETTLE_TIME
    )

    ser.reset_input_buffer()

    print(
        "Waiting for final scale reading..."
    )
    print(
        "If required, press PRINT on the KERN scale."
    )

    current_weight = read_kern_weight(
        ser
    )

    liquid_weight = (
        current_weight
        - tare_weight
    )

    if liquid_weight < 0:
        liquid_weight = 0.0

    print()
    print("======================================")
    print("STEP 5 SCALE RESULT")
    print("======================================")

    print(
        f"Starting Cup 2 weight = "
        f"{tare_weight:.1f} g"
    )

    print(
        f"Final scale weight = "
        f"{current_weight:.1f} g"
    )

    print(
        f"Liquid added = "
        f"{liquid_weight:.1f} g"
    )

    return liquid_weight


def read_fresh_kern_weight(
    ser,
    timeout_seconds=SCALE_READ_TIMEOUT
):
    """
    Read one fresh numeric KERN scale value with a timeout.

    The serial input buffer is cleared first so an old value
    is not accidentally used for robot control.

    Returns:
        float weight in grams, or None if no valid reading
        arrives before timeout.
    """

    ser.reset_input_buffer()

    deadline = (
        time.monotonic()
        + timeout_seconds
    )

    while time.monotonic() < deadline:

        data = ser.readline()

        if not data:
            continue

        text_line = (
            data.decode(
                "ascii",
                errors="ignore"
            )
            .strip()
        )

        match = re.search(
            r"[-+]?\d+(?:\.\d+)?",
            text_line
        )

        if not match:
            continue

        return float(
            match.group()
        )

    return None


def get_liquid_added_weight(
    ser,
    tare_weight
):
    """
    Read a fresh scale value and return liquid mass added
    relative to the saved Cup-2 starting weight.
    """

    current_weight = read_fresh_kern_weight(
        ser
    )

    if current_weight is None:

        return None, None

    liquid_weight = (
        current_weight
        - tare_weight
    )

    if liquid_weight < 0:
        liquid_weight = 0.0

    return (
        current_weight,
        liquid_weight
    )


def get_liquid_and_target_volume_from_user():
    """
    Ask the user to select the liquid and enter the desired
    dispensing volume in mL.

    Returns:
        liquid_key
        liquid_display_name
        density_g_per_ml
        target_volume_ml
        target_weight_g
    """

    print()
    print("======================================")
    print("SELECT LIQUID TYPE")
    print("======================================")
    print("1 = Water")
    print("2 = Mustard Oil")
    print("3 = Dishwashing Liquid")
    print("4 = Honey")
    print()

    liquid_aliases = {
        "1": "water",
        "water": "water",

        "2": "mustard_oil",
        "mustard": "mustard_oil",
        "mustard oil": "mustard_oil",
        "mustard_oil": "mustard_oil",

        "3": "dishwashing_liquid",
        "dishwashing": "dishwashing_liquid",
        "dishwashing liquid": "dishwashing_liquid",
        "dish-wash liquid": "dishwashing_liquid",
        "dish wash liquid": "dishwashing_liquid",
        "dishwashing_liquid": "dishwashing_liquid",

        "4": "honey",
        "honey": "honey",
    }

    while True:

        selection = input(
            "Enter liquid number or name: "
        ).strip().lower()

        liquid_key = liquid_aliases.get(
            selection
        )

        if liquid_key is None:

            print(
                "Invalid liquid selection. "
                "Please choose 1, 2, 3 or 4."
            )
            continue

        break

    liquid_name = (
        LIQUID_DISPLAY_NAMES[
            liquid_key
        ]
    )

    density = float(
        LIQUID_DENSITIES[
            liquid_key
        ]
    )

    while True:

        value = input(
            "Enter desired volume in mL "
            "(example: 25): "
        ).strip()

        try:

            target_volume_ml = float(
                value
            )

        except ValueError:

            print(
                "Please enter a numeric volume."
            )
            continue

        if target_volume_ml <= 0:

            print(
                "Volume must be greater than 0 mL."
            )
            continue

        break

    target_weight_g = (
        target_volume_ml
        * density
    )

    print()
    print("======================================")
    print("VOLUME -> MASS CONVERSION")
    print("======================================")

    print(
        f"Liquid = {liquid_name}"
    )

    print(
        f"Experimental density = "
        f"{density:.3f} g/mL"
    )

    print(
        f"Desired volume = "
        f"{target_volume_ml:.2f} mL"
    )

    print(
        f"Calculated target mass = "
        f"{target_weight_g:.2f} g"
    )

    return (
        liquid_key,
        liquid_name,
        density,
        target_volume_ml,
        target_weight_g
    )


def get_horizontal_reach(target_x, target_y):
    """
    Horizontal distance from the robot base in the XY plane.
    """
    return float(
        np.sqrt(
            (target_x ** 2)
            + (target_y ** 2)
        )
    )


def get_theta_x(target_x, target_y):
    """
    Select Theta-X according to experimentally measured
    horizontal arm reach, rather than X alone.

    Physical test results showed that diagonal positions with
    larger |Y| can require a lower Theta-X even when X itself
    is not especially large.
    """

    reach = get_horizontal_reach(
        target_x,
        target_y
    )

    if reach <= 0.735:
        return 172.7

    elif reach <= 0.755:
        return 168.5

    elif reach <= 0.785:
        return 165.0

    else:
        return 163.0


def correct_gripper_centre(raw_x, raw_y):

    # Original G1-G6 position-dependent correction
    base_y_correction = (
        0.0290264
        + 0.0467883 * raw_y
        + 0.2894850 * (raw_y ** 2)
    )

    # Latest A-D fine calibration
    fine_x_correction = 0.000
    fine_y_correction = 0.012

    corrected_x = (
        raw_x
        + fine_x_correction
    )

    corrected_y = (
        raw_y
        + base_y_correction
        + fine_y_correction
    )

    total_y_correction = (
        base_y_correction
        + fine_y_correction
    )

    return (
        corrected_x,
        corrected_y,
        total_y_correction
    )


# ==========================================================
# STABILITY SETTINGS
# ==========================================================

WINDOW_SIZE = 15

MIN_DETECTIONS = 8

# Maximum X/Y variation allowed
MAX_POSITION_ERROR = 0.012

recent_targets = deque(
    maxlen=WINDOW_SIZE
)


# ==========================================================
# 2. CAMERA CALIBRATION POINTS
# ==========================================================

camera_points = np.array([
    [198, 303],   # P1
    [217, 238],   # P2
    [230, 184],   # P3

    [339, 305],   # P4
    [332, 235],   # P5
    [329, 183],   # P6

    [478, 300],   # P7
    [448, 231],   # P8
    [428, 183]    # P9
], dtype=np.float32)


robot_points = np.array([
    [0.562, -0.161],   # P1
    [0.559,  0.016],   # P2
    [0.554,  0.213],   # P3

    [0.702, -0.161],   # P4
    [0.695,  0.024],   # P5
    [0.691,  0.216],   # P6

    [0.823, -0.150],   # P7
    [0.821,  0.039],   # P8
    [0.813,  0.219]    # P9
], dtype=np.float32)


# ==========================================================
# 3. WORKSPACE BOUNDARY
# ==========================================================

workspace_polygon = np.array([
    [120, 165],   # top-left
    [520, 165],   # top-right
    [570, 375],   # bottom-right
    [80,  375]    # bottom-left
], dtype=np.int32)

workspace_polygon = workspace_polygon.reshape(
    (-1, 1, 2)
)


# ==========================================================
# 4. HOMOGRAPHY
# ==========================================================

H, status = cv2.findHomography(
    camera_points,
    robot_points
)

if H is None:
    print("ERROR: Camera calibration failed")
    sys.exit(1)


# ==========================================================
# 5. PIXEL -> ROBOT
# ==========================================================

def pixel_to_robot(u, v):

    pixel = np.array(
        [[[u, v]]],
        dtype=np.float32
    )

    robot = cv2.perspectiveTransform(
        pixel,
        H
    )

    robot_x = float(
        robot[0][0][0]
    )

    robot_y = float(
        robot[0][0][1]
    )

    # NEW fixed correction from the September recalibration.
    # Derived from validation points A-D and checked with E-F.
    #
    # Old position-dependent correction equations are NOT used.
    robot_x += 0.0290
    robot_y += 0.0135

    return robot_x, robot_y


# ==========================================================
# 6. STABLE TARGET FUNCTION
# ==========================================================

def get_stable_target():

    valid_targets = [
        t
        for t in recent_targets
        if t is not None
    ]

    if len(valid_targets) < MIN_DETECTIONS:
        return None

    xs = np.array([
        target[0]
        for target in valid_targets
    ])

    ys = np.array([
        target[1]
        for target in valid_targets
    ])

    median_x = np.median(xs)
    median_y = np.median(ys)

    x_error = np.max(
        np.abs(xs - median_x)
    )

    y_error = np.max(
        np.abs(ys - median_y)
    )

    if (
        x_error <= MAX_POSITION_ERROR
        and
        y_error <= MAX_POSITION_ERROR
    ):

        return (
            float(median_x),
            float(median_y)
        )

    return None


# ==========================================================
# 7. KINOVA ACTION NOTIFICATION
# ==========================================================

def check_for_end_or_abort(e):

    def check(notification):

        if notification.action_event == Base_pb2.ACTION_END:

            print("Robot action completed.")
            e.set()

        elif notification.action_event == Base_pb2.ACTION_ABORT:

            print("Robot action aborted!")
            e.set()

    return check


# ==========================================================
# 8. MOVE TO SAVED HOME ACTION
# ==========================================================

def move_home(base):

    print()
    print("Looking for saved Home action...")

    action_type = Base_pb2.RequestedActionType()

    action_type.action_type = (
        Base_pb2.REACH_JOINT_ANGLES
    )

    action_list = base.ReadAllActions(
        action_type
    )

    home_handle = None

    for action in action_list.action_list:

        if action.name == "Home":

            home_handle = action.handle
            break


    if home_handle is None:

        print("ERROR: Saved action 'Home' not found.")
        return False


    print("Moving robot to Home...")


    e = threading.Event()

    notification_handle = (
        base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
    )


    base.ExecuteActionFromReference(
        home_handle
    )


    finished = e.wait(30)


    base.Unsubscribe(
        notification_handle
    )


    if finished:

        print("Robot reached Home.")
        return True

    else:

        print("Home movement timed out.")
        return False


# ==========================================================
# 9. DIRECT GRIPPER COMMAND
# ==========================================================

def move_gripper(base, position):

    print()
    print(f"Gripper command = {position:.2f}")

    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION

    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position

    base.SendGripperCommand(
        gripper_command
    )

    time.sleep(2)


# ==========================================================
# 10. REUSABLE CARTESIAN MOVE
# ==========================================================

def move_to_pose(
    base,
    x,
    y,
    z,
    theta_x,
    theta_y,
    theta_z,
    action_name="Cartesian Move"
):

    action = Base_pb2.Action()

    action.name = action_name
    action.application_data = ""

    pose = action.reach_pose.target_pose

    pose.x = x
    pose.y = y
    pose.z = z

    pose.theta_x = theta_x
    pose.theta_y = theta_y
    pose.theta_z = theta_z


    e = threading.Event()

    notification_handle = (
        base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
    )


    print()
    print(
        f"{action_name}: "
        f"X={x:.3f}, Y={y:.3f}, Z={z:.3f}, "
        f"ThetaX={theta_x:.1f}"
    )


    base.ExecuteAction(
        action
    )


    finished = e.wait(30)


    base.Unsubscribe(
        notification_handle
    )


    if finished:

        print(
            f"{action_name} completed."
        )

        return True

    else:

        print(
            f"{action_name} timed out."
        )

        return False


# ==========================================================
# 11. MOVE ABOVE DETECTED CUP
# ==========================================================

def move_above_cup(
    base,
    target_x,
    target_y
):

    reach = get_horizontal_reach(
        target_x,
        target_y
    )

    theta_x = get_theta_x(
        target_x,
        target_y
    )

    print()
    print("======================================")
    print("MOVING ABOVE DETECTED CUP")
    print("======================================")

    print(f"Target X = {target_x:.3f} m")
    print(f"Target Y = {target_y:.3f} m")
    print(f"Safe Z   = {SAFE_Z:.3f} m")
    print(f"XY Reach = {reach:.3f} m")
    print(f"Theta X  = {theta_x:.1f} (reach-based)")
    print(f"Theta Y  = {THETA_Y:.1f}")
    print(f"Theta Z  = {THETA_Z:.1f}")

    return move_to_pose(
        base,
        target_x,
        target_y,
        SAFE_Z,
        theta_x,
        THETA_Y,
        THETA_Z,
        "Move Above Detected Cup"
    )


# ==========================================================
# 11B. MOVE ONLY ACTUATOR 7
# ==========================================================

def move_only_actuator_7_to(
    base,
    target_actuator_7,
    action_name="Actuator 7 Move"
):
    """
    Move ONLY Actuator 7 to an absolute target angle.

    The robot's CURRENT measured values for Actuators 1-6 are
    read immediately before the action and reused unchanged.
    """

    measured = base.GetMeasuredJointAngles()

    if len(measured.joint_angles) < 7:

        print(
            "ERROR: Could not read all 7 joint angles."
        )

        return False

    current_angles = {}

    for joint in measured.joint_angles:

        current_angles[
            int(joint.joint_identifier)
        ] = float(joint.value)

    required_ids = list(range(7))

    if not all(
        joint_id in current_angles
        for joint_id in required_ids
    ):

        print(
            "ERROR: Joint-angle feedback is incomplete."
        )

        return False

    current_actuator_7 = current_angles[6]

    print()
    print(
        f"{action_name}: "
        f"Actuator 7 "
        f"{current_actuator_7:.3f} -> "
        f"{target_actuator_7:.3f} deg"
    )

    action = Base_pb2.Action()

    action.name = action_name
    action.application_data = ""

    for joint_id in range(7):

        joint_angle = (
            action
            .reach_joint_angles
            .joint_angles
            .joint_angles
            .add()
        )

        joint_angle.joint_identifier = joint_id

        if joint_id == 6:

            # ONLY Actuator 7 changes.
            joint_angle.value = float(
                target_actuator_7
            )

        else:

            # Keep Actuators 1-6 at their current values.
            joint_angle.value = float(
                current_angles[joint_id]
            )

    e = threading.Event()

    result = {
        "aborted": False
    }

    def joint_notification(notification):

        if (
            notification.action_event
            == Base_pb2.ACTION_END
        ):

            e.set()

        elif (
            notification.action_event
            == Base_pb2.ACTION_ABORT
        ):

            print(
                f"{action_name} ABORTED."
            )

            result["aborted"] = True
            e.set()

    notification_handle = (
        base.OnNotificationActionTopic(
            joint_notification,
            Base_pb2.NotificationOptions()
        )
    )

    base.ExecuteAction(
        action
    )

    finished = e.wait(30)

    base.Unsubscribe(
        notification_handle
    )

    if not finished:

        print(
            f"{action_name} timed out."
        )

        return False

    if result["aborted"]:

        return False

    final_measured = (
        base.GetMeasuredJointAngles()
    )

    final_actuator_7 = None

    for joint in final_measured.joint_angles:

        if int(joint.joint_identifier) == 6:

            final_actuator_7 = float(
                joint.value
            )

            break

    if final_actuator_7 is not None:

        print(
            f"Measured Actuator 7 = "
            f"{final_actuator_7:.3f} deg"
        )

    return True


# ==========================================================
# 11C. AUTOMATIC CALIBRATED POUR
# ==========================================================

def automatic_full_pour(base):
    """
    Automatic full-pour sequence using the experimentally
    measured Actuator 7 angles:

        current upright
            ->
        114.60 deg  (water starts pouring)
            ->
        230.00 deg  (full / maximum pour)
            ->
        original upright angle

    Actuators 1-6 remain unchanged.
    """

    print()
    print("======================================")
    print("STEP 4: AUTOMATIC FULL POUR")
    print("======================================")

    # Read and save the exact current upright Actuator 7 angle.
    measured = base.GetMeasuredJointAngles()

    if len(measured.joint_angles) < 7:

        print(
            "ERROR: Could not read current joint angles."
        )

        return False

    original_actuator_7 = None

    for joint in measured.joint_angles:

        if int(joint.joint_identifier) == 6:

            original_actuator_7 = float(
                joint.value
            )

            break

    if original_actuator_7 is None:

        print(
            "ERROR: Could not read Actuator 7."
        )

        return False

    print()
    print(
        f"Upright Actuator 7 = "
        f"{original_actuator_7:.3f} deg"
    )

    print(
        f"Pour starts at = "
        f"{POUR_START_ACTUATOR_7:.2f} deg"
    )

    print(
        f"Full pour = "
        f"{POUR_FULL_ACTUATOR_7:.2f} deg"
    )

    print()
    print(
        "Actuators 1-6 will remain unchanged."
    )
    print(
        "Gripper will remain CLOSED."
    )

    # --------------------------------------------------
    # STAGE 1 - Move to first-flow angle
    # --------------------------------------------------

    print()
    print("Stage 1: Moving to first-flow angle...")

    start_ok = move_only_actuator_7_to(
        base,
        POUR_START_ACTUATOR_7,
        "Move To Pour Start"
    )

    if not start_ok:

        print(
            "Could not reach pour-start angle."
        )

        return False

    print(
        "Pour-start angle reached."
    )

    time.sleep(
        POUR_START_PAUSE
    )

    # --------------------------------------------------
    # STAGE 2 - Continue to full-pour angle
    # --------------------------------------------------

    print()
    print("Stage 2: Moving to full-pour angle...")

    full_ok = move_only_actuator_7_to(
        base,
        POUR_FULL_ACTUATOR_7,
        "Move To Full Pour"
    )

    if not full_ok:

        print(
            "Could not reach full-pour angle."
        )

        return False

    print()
    print(
        "FULL POUR POSITION REACHED."
    )

    print(
        f"Holding for "
        f"{POUR_FULL_HOLD:.1f} seconds..."
    )

    time.sleep(
        POUR_FULL_HOLD
    )

    # --------------------------------------------------
    # STAGE 3 - Return cup upright
    # --------------------------------------------------

    print()
    print("Stage 3: Returning cup upright...")

    return_ok = move_only_actuator_7_to(
        base,
        original_actuator_7,
        "Return Cup Upright"
    )

    if not return_ok:

        print(
            "WARNING: Could not return to upright angle."
        )

        return False

    print()
    print("======================================")
    print("STEP 4 POUR COMPLETE")
    print("======================================")

    print(
        "Cup has returned to its original "
        "Actuator 7 orientation."
    )

    print(
        "Gripper remains CLOSED."
    )

    return True


# ==========================================================
# STEP 6 - CLOSED-LOOP TARGET-WEIGHT POUR
# ==========================================================

def target_weight_pour(
    base,
    ser,
    tare_weight,
    target_weight,
    liquid_name,
    liquid_density,
    target_volume_ml
):
    """
    Conservative pulse-and-weigh target-mass control.

    IMPORTANT:
    This replaces continuous pouring because the first
    closed-loop test overshot:
        target = 25 g
        final  = 50 g

    Water / mustard-oil cycle:
        read weight
        -> brief Actuator-7 pour pulse
        -> return upright
        -> wait for scale settling
        -> read new weight

    Dishwashing-liquid / honey cycle:
        move to Actuator-7 angle
        -> hold at that SAME angle
        -> repeatedly monitor KERN scale
        -> while measurable flow continues, keep holding
        -> when flow stops, return upright
        -> increase angle by 2 degrees
        -> repeat until target is reached

    Actuators 1-6 remain unchanged.
    """

    print()
    print("======================================")
    print("STEP 6B: PULSE-AND-WEIGH POUR")
    print("======================================")

    print(
        f"Liquid type = "
        f"{liquid_name}"
    )

    print(
        f"Desired volume = "
        f"{target_volume_ml:.2f} mL"
    )

    print(
        f"Liquid density = "
        f"{liquid_density:.3f} g/mL"
    )

    print(
        f"Calculated target mass = "
        f"{target_weight:.2f} g"
    )

    print(
        f"Early-stop margin = "
        f"{EARLY_STOP_MARGIN_G:.1f} g"
    )

    # ------------------------------------------------------
    # Save exact upright Actuator-7 angle
    # ------------------------------------------------------

    measured = base.GetMeasuredJointAngles()

    if len(measured.joint_angles) < 7:

        print(
            "ERROR: Could not read robot joint angles."
        )

        return False

    original_actuator_7 = None

    for joint in measured.joint_angles:

        if int(joint.joint_identifier) == 6:

            original_actuator_7 = float(
                joint.value
            )

            break

    if original_actuator_7 is None:

        print(
            "ERROR: Could not read Actuator 7."
        )

        return False

    print(
        f"Saved upright Actuator 7 = "
        f"{original_actuator_7:.3f} deg"
    )

    # ------------------------------------------------------
    # Verify live scale data before moving
    # ------------------------------------------------------

    print()
    print(
        "Checking live KERN scale feedback..."
    )

    current_total, liquid_weight = (
        get_liquid_added_weight(
            ser,
            tare_weight
        )
    )

    if current_total is None:

        print()
        print("======================================")
        print("NO LIVE SCALE DATA")
        print("======================================")

        print(
            "Robot will NOT start pouring."
        )

        return False

    print(
        f"Initial liquid = "
        f"{liquid_weight:.1f} g"
    )

    if liquid_weight >= target_weight:

        print(
            "Target is already reached."
        )

        return True

    # ------------------------------------------------------
    # Pulse controller state
    # ------------------------------------------------------

    pulse_angle = float(
        PULSE_START_ANGLE
    )

    previous_liquid = float(
        liquid_weight
    )

    pulse_number = 0
    stopped_early = False

    # Final <=3 g fine-angle mode state.
    # The -4 degree back-off is applied once only.
    final_fine_angle_mode = False

    # Stop after repeated no-flow pulses at the maximum
    # search angle instead of pulsing indefinitely.
    max_angle_no_flow_count = 0
    MAX_ANGLE_NO_FLOW_PULSES = 3

    # ------------------------------------------------------
    # DISPENSE -> UPRIGHT -> WEIGH cycles
    # ------------------------------------------------------

    while pulse_number < MAX_POUR_PULSES:

        remaining = (
            target_weight
            - previous_liquid
        )

        # --------------------------------------------------
        # FINAL <=3 g FINE-ANGLE MODE - ALL LIQUIDS
        # --------------------------------------------------
        # When 3 g or less remains, back off the CURRENT
        # angle by 4 degrees ONCE. After that, any required
        # upward adjustment uses +1.5 degrees instead of +4.
        if (
            remaining <= FINAL_FINE_REMAINING_G
            and
            not final_fine_angle_mode
        ):

            old_angle_before_backoff = pulse_angle

            pulse_angle = max(
                pulse_angle - FINAL_FINE_ANGLE_BACKOFF,
                PULSE_MIN_ANGLE
            )

            final_fine_angle_mode = True

            print()
            print("======================================")
            print("FINAL FINE-ANGLE MODE ACTIVATED")
            print("======================================")
            print(f"Remaining mass = {remaining:.2f} g")
            print(
                f"Angle backed off once: "
                f"{old_angle_before_backoff:.2f} -> "
                f"{pulse_angle:.2f} deg"
            )
            print(
                f"Further angle increases will use "
                f"+{FINAL_FINE_ANGLE_UP_STEP:.1f} deg steps."
            )

        print()
        print("--------------------------------------")
        print(
            f"Before pulse {pulse_number + 1}: "
            f"{previous_liquid:.1f} g poured | "
            f"{remaining:.1f} g remaining"
        )

        # Stop slightly early to compensate for final drops.
        if (
            remaining
            <= EARLY_STOP_MARGIN_G
        ):

            print(
                "Within early-stop margin. "
                "No more pulse will be made."
            )

            stopped_early = True
            break

        # ----------------------------------------------
        # Select pulse strength
        # ----------------------------------------------

        # Keep the current adaptive angle in both modes.
        # Fine mode reduces pulse duration only; it does not
        # force the angle down to 118 deg.

        commanded_angle = min(
            max(
                pulse_angle,
                PULSE_MIN_ANGLE
            ),
            PULSE_MAX_ANGLE
        )

        # --------------------------------------------------
        # Select hold time according to liquid type.
        #
        # Water and mustard oil keep the original validated
        # timings. Only dishwashing liquid and honey receive
        # longer holds because they flow more slowly.
        # --------------------------------------------------

        liquid_key_for_control = (
            liquid_name
            .strip()
            .lower()
        )

        if remaining <= FINE_REMAINING_G:

            mode = "FINE"

            if liquid_key_for_control == "dishwashing liquid":

                pulse_hold = (
                    DISHWASH_FINE_PULSE_HOLD
                )

            elif liquid_key_for_control == "honey":

                pulse_hold = (
                    HONEY_FINE_PULSE_HOLD
                )

            else:

                # Water / Mustard Oil
                pulse_hold = (
                    FINE_PULSE_HOLD
                )

        else:

            mode = "COARSE"

            if liquid_key_for_control == "dishwashing liquid":

                pulse_hold = (
                    DISHWASH_COARSE_PULSE_HOLD
                )

            elif liquid_key_for_control == "honey":

                pulse_hold = (
                    HONEY_COARSE_PULSE_HOLD
                )

            else:

                # Water / Mustard Oil
                pulse_hold = (
                    COARSE_PULSE_HOLD
                )

        print(
            f"Pulse mode = {mode}"
        )

        print(
            f"Pulse angle = "
            f"{commanded_angle:.2f} deg"
        )

        print(
            f"Pulse hold = "
            f"{pulse_hold:.2f} s"
        )

        if liquid_key_for_control == "dishwashing liquid":

            print(
                "Viscous-liquid timing: "
                "extended hold for Dishwashing Liquid."
            )

        elif liquid_key_for_control == "honey":

            print(
                "Viscous-liquid timing: "
                "extended hold for Honey."
            )

        # ----------------------------------------------
        # Briefly tilt to dispensing angle
        # ----------------------------------------------

        pulse_ok = move_only_actuator_7_to(
            base,
            commanded_angle,
            f"Target Pour Pulse {pulse_number + 1}"
        )

        if not pulse_ok:

            print(
                "Pulse movement failed."
            )

            # Attempt upright recovery.
            move_only_actuator_7_to(
                base,
                original_actuator_7,
                "Recovery - Return Upright"
            )

            return False

        # ----------------------------------------------
        # HOLD / FLOW MONITOR
        # ----------------------------------------------

        viscous_liquid = (
            liquid_key_for_control
            in (
                "dishwashing liquid",
                "honey"
            )
        )

        viscous_flow_ended = False
        viscous_max_hold_reached = False

        if not viscous_liquid:

            # Water and mustard oil keep the original
            # short pulse behavior unchanged.
            time.sleep(
                pulse_hold
            )

        else:

            print()
            print("======================================")
            print("VISCOUS LIQUID FLOW-HOLD ACTIVE")
            print("======================================")
            print(
                "Robot will stay at the current angle "
                "while measurable liquid continues to flow."
            )

            if liquid_key_for_control == "dishwashing liquid":

                minimum_hold_time = (
                    DISHWASH_MIN_HOLD_TIME
                )

                flow_check_interval = (
                    DISHWASH_FLOW_CHECK_INTERVAL
                )

                maximum_hold_time = (
                    DISHWASH_MAX_HOLD_TIME
                )

            else:

                minimum_hold_time = (
                    HONEY_MIN_HOLD_TIME
                )

                flow_check_interval = (
                    HONEY_FLOW_CHECK_INTERVAL
                )

                maximum_hold_time = (
                    HONEY_MAX_HOLD_TIME
                )

            print(
                f"Minimum hold before flow checks = "
                f"{minimum_hold_time:.2f} s"
            )

            print(
                f"Flow check interval = "
                f"{flow_check_interval:.2f} s"
            )

            print(
                f"Maximum hold safety limit = "
                f"{maximum_hold_time:.2f} s"
            )

            # Give the viscous liquid enough time to start
            # moving through the lip of the source cup.
            time.sleep(
                minimum_hold_time
            )

            hold_start_time = (
                time.monotonic()
            )

            last_live_liquid = float(
                previous_liquid
            )

            no_flow_count = 0

            while True:

                # Check whether maximum hold safety time
                # has been reached.
                elapsed_hold = (
                    time.monotonic()
                    - hold_start_time
                    + minimum_hold_time
                )

                if elapsed_hold >= maximum_hold_time:

                    print()
                    print(
                        f"Maximum viscous hold time "
                        f"{maximum_hold_time:.2f} s reached."
                    )

                    print(
                        "Returning upright before continuing."
                    )

                    viscous_max_hold_reached = True
                    break

                time.sleep(
                    flow_check_interval
                )

                live_total, live_liquid = (
                    get_liquid_added_weight(
                        ser,
                        tare_weight
                    )
                )

                if live_total is None:

                    print()
                    print("======================================")
                    print("LIVE SCALE FEEDBACK LOST")
                    print("======================================")
                    print(
                        "Attempting immediate upright recovery."
                    )

                    move_only_actuator_7_to(
                        base,
                        original_actuator_7,
                        "Recovery - Return Upright"
                    )

                    return False

                live_gain = (
                    live_liquid
                    - last_live_liquid
                )

                if live_gain < 0:
                    live_gain = 0.0

                live_remaining = (
                    target_weight
                    - live_liquid
                )

                print(
                    f"Live hold reading: "
                    f"{live_liquid:.1f} g poured | "
                    f"gain = {live_gain:.1f} g | "
                    f"remaining = "
                    f"{max(live_remaining, 0.0):.1f} g"
                )

                # Stop holding as soon as the target is
                # inside the existing early-stop margin.
                if (
                    live_remaining
                    <= EARLY_STOP_MARGIN_G
                ):

                    print(
                        "Target is inside early-stop margin "
                        "during viscous flow."
                    )

                    print(
                        "Returning upright now."
                    )

                    stopped_early = True
                    break

                # Flow detected:
                # keep the SAME angle and continue holding.
                if (
                    live_gain
                    >= VISCOUS_FLOW_GAIN_THRESHOLD_G
                ):

                    no_flow_count = 0

                    print(
                        "Flow detected -> keeping same angle."
                    )

                else:

                    no_flow_count += 1

                    print(
                        f"No measurable new flow: "
                        f"{no_flow_count}/"
                        f"{VISCOUS_NO_FLOW_CONFIRMATIONS}"
                    )

                    if (
                        no_flow_count
                        >= VISCOUS_NO_FLOW_CONFIRMATIONS
                    ):

                        print(
                            "Flow has stopped at this angle."
                        )

                        print(
                            "Robot will return upright, "
                            "then increase the angle."
                        )

                        viscous_flow_ended = True
                        break

                last_live_liquid = float(
                    live_liquid
                )

        # ----------------------------------------------
        # Return upright
        # ----------------------------------------------

        upright_ok = move_only_actuator_7_to(
            base,
            original_actuator_7,
            f"Pulse {pulse_number + 1} - Return Upright"
        )

        if not upright_ok:

            print(
                "WARNING: Could not confirm upright return."
            )

            return False

        pulse_number += 1

        # ----------------------------------------------
        # Wait for falling liquid + scale to settle
        # ----------------------------------------------

        print(
            f"Waiting {PULSE_SETTLE_TIME:.2f} s "
            "for liquid and scale to settle..."
        )

        time.sleep(
            PULSE_SETTLE_TIME
        )

        # ----------------------------------------------
        # Fresh scale reading
        # ----------------------------------------------

        current_total, new_liquid = (
            get_liquid_added_weight(
                ser,
                tare_weight
            )
        )

        if current_total is None:

            print()
            print("======================================")
            print("SCALE FEEDBACK LOST")
            print("======================================")

            print(
                "Robot is upright. "
                "Pouring has stopped."
            )

            return False

        pulse_gain = (
            new_liquid
            - previous_liquid
        )

        if pulse_gain < 0:
            pulse_gain = 0.0

        if (
            commanded_angle >= PULSE_MAX_ANGLE
            and
            pulse_gain < MIN_USEFUL_PULSE_GAIN_G
        ):

            max_angle_no_flow_count += 1

        else:

            max_angle_no_flow_count = 0

        final_remaining = (
            target_weight
            - new_liquid
        )

        print(
            f"After pulse {pulse_number}: "
            f"liquid = {new_liquid:.1f} g | "
            f"pulse gain = {pulse_gain:.1f} g | "
            f"remaining = "
            f"{max(final_remaining, 0.0):.1f} g"
        )

        if max_angle_no_flow_count >= MAX_ANGLE_NO_FLOW_PULSES:

            print()
            print("======================================")
            print("NO FLOW AT MAXIMUM SEARCH ANGLE")
            print("======================================")
            print(
                f"No useful liquid flow was detected for "
                f"{MAX_ANGLE_NO_FLOW_PULSES} consecutive pulses "
                f"at {PULSE_MAX_ANGLE:.1f} deg."
            )
            print(
                "Robot is upright. Pouring will stop."
            )

            previous_liquid = float(
                new_liquid
            )

            break

        # ----------------------------------------------
        # Target reached
        # ----------------------------------------------

        if new_liquid >= target_weight:

            previous_liquid = (
                new_liquid
            )

            print(
                "Target threshold reached. "
                "No additional pulse."
            )

            break

        # ----------------------------------------------
        # Adapt the NEXT pulse angle from measured gain
        # ----------------------------------------------

        if viscous_liquid:

            # For dishwashing liquid and honey:
            # once flow stops at the current angle, increase
            # the angle by 2 degrees for the next hold cycle.
            #
            # Do NOT reduce angle because of a large pulse gain;
            # the viscous controller now uses flow/no-flow
            # behavior rather than pulse size for angle changes.

            if viscous_flow_ended:

                old_pulse_angle = (
                    pulse_angle
                )

                angle_up_step = (
                    FINAL_FINE_ANGLE_UP_STEP
                    if final_fine_angle_mode
                    else PULSE_ANGLE_UP_STEP
                )

                pulse_angle = min(
                    pulse_angle
                    + angle_up_step,
                    PULSE_MAX_ANGLE
                )

                if pulse_angle > old_pulse_angle:

                    print(
                        f"Viscous flow stopped: "
                        f"next angle increased by "
                        f"{angle_up_step:.1f} deg to "
                        f"{pulse_angle:.2f} deg."
                    )

                else:

                    print(
                        f"Viscous flow stopped, but maximum "
                        f"search angle {PULSE_MAX_ANGLE:.2f} deg "
                        f"has been reached."
                    )

            elif viscous_max_hold_reached:

                print(
                    f"Maximum hold time reached while using "
                    f"{pulse_angle:.2f} deg."
                )

                print(
                    "Keeping the same angle for the next "
                    "control cycle."
                )

            else:

                # This normally occurs only when the live
                # scale brought the target into the early-stop
                # region; the main loop will stop on the next
                # target check.
                print(
                    f"Keeping viscous-liquid angle at "
                    f"{pulse_angle:.2f} deg."
                )

        else:

            # Original water / mustard-oil adaptive controller.
            if pulse_gain < MIN_USEFUL_PULSE_GAIN_G:

                old_pulse_angle = pulse_angle

                angle_up_step = (
                    FINAL_FINE_ANGLE_UP_STEP
                    if final_fine_angle_mode
                    else PULSE_ANGLE_UP_STEP
                )

                pulse_angle = min(
                    pulse_angle
                    + angle_up_step,
                    PULSE_MAX_ANGLE
                )

                if pulse_angle > old_pulse_angle:

                    print(
                        f"Very small pulse gain: next angle "
                        f"increased by {angle_up_step:.1f} deg to "
                        f"{pulse_angle:.2f} deg."
                    )

                else:

                    print(
                        f"Very small pulse gain, but maximum "
                        f"search angle {PULSE_MAX_ANGLE:.2f} deg "
                        f"has been reached."
                    )

            elif (
                pulse_gain >= LARGE_PULSE_GAIN_G
                and
                not final_fine_angle_mode
            ):

                pulse_angle = max(
                    pulse_angle
                    - PULSE_ANGLE_DOWN_STEP,
                    PULSE_MIN_ANGLE
                )

                print(
                    f"Large pulse gain: next angle "
                    f"reduced to {pulse_angle:.2f} deg."
                )

            else:

                print(
                    f"Useful pulse gain detected: keeping "
                    f"next angle at {pulse_angle:.2f} deg."
                )

        previous_liquid = float(
            new_liquid
        )

    # ------------------------------------------------------
    # Final settled verification
    # ------------------------------------------------------

    print()
    print(
        f"Final verification: waiting "
        f"{FINAL_WEIGHT_SETTLE_TIME:.1f} s..."
    )

    time.sleep(
        FINAL_WEIGHT_SETTLE_TIME
    )

    final_total, final_liquid = (
        get_liquid_added_weight(
            ser,
            tare_weight
        )
    )

    if final_total is None:

        print(
            "Could not obtain final scale reading."
        )

        return False

    error = (
        final_liquid
        - target_weight
    )

    print()
    print("======================================")
    print("PULSE-AND-WEIGH RESULT")
    print("======================================")

    final_volume_ml = (
        final_liquid
        / liquid_density
    )

    volume_error_ml = (
        final_volume_ml
        - target_volume_ml
    )

    print(
        f"Liquid type = "
        f"{liquid_name}"
    )

    print(
        f"Target volume = "
        f"{target_volume_ml:.2f} mL"
    )

    print(
        f"Target mass = "
        f"{target_weight:.2f} g"
    )

    print(
        f"Final liquid mass = "
        f"{final_liquid:.1f} g"
    )

    print(
        f"Mass error = "
        f"{error:+.2f} g"
    )

    print(
        f"Estimated final volume = "
        f"{final_volume_ml:.2f} mL"
    )

    print(
        f"Volume error = "
        f"{volume_error_ml:+.2f} mL"
    )

    print(
        f"Number of pour pulses = "
        f"{pulse_number}"
    )

    print(
        f"Last adaptive pulse angle = "
        f"{pulse_angle:.2f} deg"
    )

    if stopped_early:

        print(
            "Controller stopped inside the "
            "early-stop margin."
        )

    return True




# ==========================================================
# FINAL RETURN SEQUENCE - PUT CUP BACK AND GO HOME
# ==========================================================

def return_cup_to_pickup_and_home(
    base,
    pickup_x,
    pickup_y
):
    """
    After a successful pour:

        1. Move the held cup back above its original pickup X/Y.
        2. Descend to the original pickup height.
        3. Open the gripper and release the cup.
        4. Lift straight back to SAFE_Z.
        5. Move the robot to the saved Home action.

    The same JSON pickup orientation is maintained while
    returning and placing the cup.
    """

    print()
    print("======================================")
    print("FINAL RETURN SEQUENCE")
    print("======================================")
    print(
        f"Returning Cup 1 to original pickup position:"
    )
    print(
        f"X = {pickup_x:.3f} m"
    )
    print(
        f"Y = {pickup_y:.3f} m"
    )

    # ------------------------------------------------------
    # Stage 1 - Move above original pickup position
    # ------------------------------------------------------

    print()
    print("Stage 1: Moving above original pickup position...")

    above_pickup_ok = move_to_pose(
        base,
        pickup_x,
        pickup_y,
        SAFE_Z,
        PICK_THETA_X,
        PICK_THETA_Y,
        PICK_THETA_Z,
        "Return Above Original Pickup"
    )

    if not above_pickup_ok:

        print()
        print(
            "WARNING: Could not move above original pickup "
            "position."
        )
        print(
            "Cup remains held. Home move will NOT be attempted."
        )

        return False

    time.sleep(1.0)

    # ------------------------------------------------------
    # Stage 2 - Descend to original pickup height
    # ------------------------------------------------------

    print()
    print("Stage 2: Descending to original pickup height...")

    descend_ok = move_to_pose(
        base,
        pickup_x,
        pickup_y,
        PICK_Z,
        PICK_THETA_X,
        PICK_THETA_Y,
        PICK_THETA_Z,
        "Return Cup To Pickup Height"
    )

    if not descend_ok:

        print()
        print(
            "WARNING: Could not descend to original pickup "
            "height."
        )
        print(
            "Cup remains held. Home move will NOT be attempted."
        )

        return False

    time.sleep(1.0)

    # ------------------------------------------------------
    # Stage 3 - Release cup
    # ------------------------------------------------------

    print()
    print("Stage 3: Opening gripper and releasing Cup 1...")

    move_gripper(
        base,
        GRIPPER_OPEN
    )

    time.sleep(1.0)

    # ------------------------------------------------------
    # Stage 4 - Lift away from released cup
    # ------------------------------------------------------

    print()
    print("Stage 4: Lifting away from released cup...")

    lift_ok = move_to_pose(
        base,
        pickup_x,
        pickup_y,
        SAFE_Z,
        PICK_THETA_X,
        PICK_THETA_Y,
        PICK_THETA_Z,
        "Lift Away After Cup Release"
    )

    if not lift_ok:

        print()
        print(
            "WARNING: Cup was released, but robot could not "
            "confirm the lift-away movement."
        )
        print(
            "Home move will NOT be attempted."
        )

        return False

    time.sleep(1.0)

    # ------------------------------------------------------
    # Stage 5 - Home
    # ------------------------------------------------------

    print()
    print("Stage 5: Moving robot Home...")

    home_ok = move_home(
        base
    )

    if not home_ok:

        print()
        print(
            "WARNING: Cup was returned successfully, "
            "but Home movement did not complete."
        )

        return False

    print()
    print("======================================")
    print("EXPERIMENT SEQUENCE COMPLETE")
    print("======================================")
    print(
        "Cup 1 returned to its original pickup position."
    )
    print(
        "Robot returned Home."
    )

    return True


# ==========================================================
# 11C. MOVE HELD CUP ABOVE DESTINATION CUP
# ==========================================================

def move_held_cup_above_destination(
    base,
    destination_x,
    destination_y
):
    """
    Move Cup 1 above Cup 2 using the FINAL experimentally
    calibrated destination corrections.

    Final repeated physical calibration:
        X correction = -0.010 m
        Y correction = -0.130 m

    These corrections are ONLY for the receiving Cup 2 /
    scale pouring position. They do not alter Cup 1 pickup.
    """

    print()
    print("======================================")
    print("MOVE ABOVE FINAL CALIBRATED CUP 2")
    print("======================================")

    # Apply the final experimentally measured destination
    # corrections.
    target_x = (
        float(destination_x)
        + DESTINATION_X_OFFSET
    )

    target_y = (
        float(destination_y)
        + DESTINATION_Y_OFFSET
    )

    target_reach = get_horizontal_reach(
        target_x,
        target_y
    )

    print(
        f"Detected Cup 2 X = {destination_x:.3f} m"
    )

    print(
        f"Detected Cup 2 Y = {destination_y:.3f} m"
    )

    print()
    print(
        f"Destination X offset = "
        f"{DESTINATION_X_OFFSET:+.3f} m"
    )

    print(
        f"Destination Y offset = "
        f"{DESTINATION_Y_OFFSET:+.3f} m"
    )

    print()
    print(
        f"Final commanded X = {target_x:.3f} m"
    )

    print(
        f"Final commanded Y = {target_y:.3f} m"
    )

    print(
        f"Target XY reach = {target_reach:.3f} m"
    )

    print(
        f"Move height Z = {SAFE_Z:.3f} m"
    )

    print(
        "Cup 1 will remain in JSON pick orientation."
    )

    # Safety check uses the ACTUAL corrected destination.
    if (
        target_x > MAX_SAFE_X
        or
        target_reach > MAX_SAFE_REACH
    ):

        print()
        print("======================================")
        print("DESTINATION IS OUTSIDE SAFE REACH")
        print("======================================")

        print(
            f"Maximum safe X = {MAX_SAFE_X:.3f} m"
        )

        print(
            f"Maximum safe XY reach = {MAX_SAFE_REACH:.3f} m"
        )

        print(
            "Robot will NOT move toward Cup 2."
        )

        return False

    print()
    print(
        "Moving held Cup 1 above final calibrated "
        "Cup-2 position..."
    )

    move_ok = move_to_pose(
        base,
        target_x,
        target_y,
        SAFE_Z,
        PICK_THETA_X,
        PICK_THETA_Y,
        PICK_THETA_Z,
        "Move Held Cup Above Final Calibrated Destination"
    )

    if move_ok:

        print()
        print("======================================")
        print("FINAL CALIBRATED DESTINATION REACHED")
        print("======================================")

        print(
            f"X = {target_x:.3f} m"
        )

        print(
            f"Y = {target_y:.3f} m"
        )

        print(
            f"Z = {SAFE_Z:.3f} m"
        )

        print(
            "Cup remains UPRIGHT."
        )

        return True

    print()
    print(
        "Movement to Cup 2 did not complete."
    )

    return False


# ==========================================================
# 11C. DETECT DESTINATION CUP AFTER PICK AND LIFT
# ==========================================================

def detect_destination_cup(
    cap,
    source_x,
    source_y
):

    print()
    print("======================================")
    print("DETECTING DESTINATION CUP")
    print("======================================")
    print("Robot is holding Cup 1.")
    print("Searching for Cup 2...")
    print("Press D when DESTINATION STABLE.")
    print("Press Q to cancel.")
    print()

    destination_targets = deque(
        maxlen=DEST_WINDOW_SIZE
    )

    # Remove a few old camera frames.
    for _ in range(10):
        cap.read()

    while True:

        ret, raw_frame = cap.read()

        if not ret:

            print("Camera frame failed.")
            return None

        results = model(
            raw_frame,
            verbose=False
        )

        frame = raw_frame.copy()

        possible_cups = []

        # ==============================================
        # LOOK FOR ALL CUPS
        # ==============================================

        for result in results:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    model.names[class_id]
                )

                if (
                    class_name != "cup"
                    or
                    confidence < CONFIDENCE_THRESHOLD
                ):
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Use the same bottom-centre point as the
                # existing pickup calibration.
                cup_u = int(
                    (x1 + x2) / 2
                )

                cup_v = int(
                    y2
                )

                # ======================================
                # WORKSPACE CHECK
                # ======================================

                workspace_distance = (
                    cv2.pointPolygonTest(
                        workspace_polygon,
                        (
                            float(cup_u),
                            float(cup_v)
                        ),
                        True
                    )
                )

                if (
                    workspace_distance
                    < -WORKSPACE_MARGIN
                ):
                    continue

                # ======================================
                # CAMERA -> ROBOT
                # ======================================

                robot_x, robot_y = (
                    pixel_to_robot(
                        cup_u,
                        cup_v
                    )
                )

                # ======================================
                # DISTANCE FROM ORIGINAL CUP 1 POSITION
                # ======================================

                distance_from_source = float(
                    np.sqrt(
                        (robot_x - source_x) ** 2
                        +
                        (robot_y - source_y) ** 2
                    )
                )

                # Ignore detections near the original
                # source-cup position.
                if (
                    distance_from_source
                    < DEST_MIN_DISTANCE
                ):
                    continue

                # ======================================
                # BASIC ROBOT REACH CHECK
                # ======================================

                target_reach = get_horizontal_reach(
                    robot_x,
                    robot_y
                )

                if (
                    robot_x > MAX_SAFE_X
                    or
                    target_reach > MAX_SAFE_REACH
                ):
                    continue

                possible_cups.append(
                    (
                        confidence,
                        robot_x,
                        robot_y,
                        x1,
                        y1,
                        x2,
                        y2,
                        cup_u,
                        cup_v
                    )
                )

        # ==============================================
        # SELECT BEST CUP 2
        # ==============================================

        if possible_cups:

            best = max(
                possible_cups,
                key=lambda item: item[0]
            )

            (
                confidence,
                destination_x,
                destination_y,
                x1,
                y1,
                x2,
                y2,
                cup_u,
                cup_v
            ) = best

            destination_targets.append(
                (
                    destination_x,
                    destination_y
                )
            )

            # Green box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Purple target point
            cv2.circle(
                frame,
                (cup_u, cup_v),
                7,
                (255, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "DESTINATION CUP",
                (
                    x1,
                    max(y1 - 35, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    f"X={destination_x:.3f} "
                    f"Y={destination_y:.3f}"
                ),
                (
                    x1,
                    max(y1 - 10, 45)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        else:

            destination_targets.append(
                None
            )

        # ==============================================
        # STABILITY CHECK
        # ==============================================

        valid_targets = [
            target
            for target in destination_targets
            if target is not None
        ]

        stable_destination = None

        if (
            len(valid_targets)
            >= DEST_MIN_DETECTIONS
        ):

            xs = np.array(
                [
                    target[0]
                    for target in valid_targets
                ]
            )

            ys = np.array(
                [
                    target[1]
                    for target in valid_targets
                ]
            )

            median_x = float(
                np.median(xs)
            )

            median_y = float(
                np.median(ys)
            )

            x_error = float(
                np.max(
                    np.abs(
                        xs - median_x
                    )
                )
            )

            y_error = float(
                np.max(
                    np.abs(
                        ys - median_y
                    )
                )
            )

            if (
                x_error <= MAX_POSITION_ERROR
                and
                y_error <= MAX_POSITION_ERROR
            ):

                stable_destination = (
                    median_x,
                    median_y
                )

        # ==============================================
        # DISPLAY STATUS
        # ==============================================

        if stable_destination is not None:

            cv2.putText(
                frame,
                "DESTINATION STABLE",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "Press D to accept",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "SEARCHING FOR CUP 2...",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.polylines(
            frame,
            [workspace_polygon],
            True,
            (255, 0, 0),
            2
        )

        cv2.imshow(
            "DESTINATION CUP DETECTION",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # Q = cancel
        if key == ord("q"):

            cv2.destroyAllWindows()

            print(
                "Destination detection cancelled."
            )

            return None

        # D = accept destination
        if (
            key == ord("d")
            and
            stable_destination is not None
        ):

            cv2.destroyAllWindows()

            print()
            print("======================================")
            print("DESTINATION CUP ACCEPTED")
            print("======================================")

            print(
                f"Destination X = "
                f"{stable_destination[0]:.3f}"
            )

            print(
                f"Destination Y = "
                f"{stable_destination[1]:.3f}"
            )

            return stable_destination


# ==========================================================
# 12. CAMERA + YOLO LOOP
# ==========================================================

def detect_cup_and_move(base):

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )


    if not cap.isOpened():

        print("Could not open Logitech BRIO")
        return


    print()
    print("======================================")
    print("BRIO + YOLO + KINOVA")
    print("======================================")

    print()
    print("NEW September camera-to-robot calibration loaded.")
    print("Workspace = updated large workspace.")
    print(f"SAFE_Z = {SAFE_Z:.3f} m.")
    print("Reach-based adaptive Theta-X enabled.")
    print("G1-G6 gripper-centre Y correction enabled.")
    print(f"Pickup Z = {PICK_Z:.3f} m.")
    print(f"Gripper close command = {GRIPPER_CLOSE:.2f}.")
    print(
        "JSON pick orientation = "
        f"({PICK_THETA_X:.2f}, "
        f"{PICK_THETA_Y:.2f}, "
        f"{PICK_THETA_Z:.2f}) deg"
    )
    print(
        f"JSON pickup X correction = "
        f"{JSON_PICK_X_OFFSET:+.3f} m"
    )
    print(
        f"Destination X correction = "
        f"{DESTINATION_X_OFFSET:+.3f} m"
    )
    print(
        f"Destination Y correction = "
        f"{DESTINATION_Y_OFFSET:+.3f} m"
    )
    print(
        "Expanded camera workspace enabled."
    )
    print(f"Maximum validated robot X = {MAX_SAFE_X:.3f} m.")
    print(f"Maximum validated XY reach = {MAX_SAFE_REACH:.3f} m.")
    print("Targets outside either movement limit will be detected but NOT executed.")
    print("Searching for cup...")
    print()
    print("When TARGET STABLE appears:")
    print("Press M = move robot above cup")
    print("Press Q = quit without moving")
    print()


    accepted_target = None

    robot_has_moved = False


    while True:

        ret, raw_frame = cap.read()

        if not ret:

            print("Camera frame failed.")
            break


        # ==================================================
        # YOLO gets clean image
        # ==================================================

        results = model(
            raw_frame,
            verbose=False
        )


        frame = raw_frame.copy()


        best_detection = None
        best_confidence = 0


        # ==================================================
        # FIND BEST CUP
        # ==================================================

        for result in results:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    model.names[class_id]
                )


                if (
                    class_name == "cup"
                    and
                    confidence >= CONFIDENCE_THRESHOLD
                    and
                    confidence > best_confidence
                ):

                    best_detection = box
                    best_confidence = confidence


        # ==================================================
        # CUP FOUND
        # ==================================================

        if best_detection is not None:

            x1, y1, x2, y2 = map(
                int,
                best_detection.xyxy[0]
            )


            # Bounding box centre
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ==================================================
            # IMPORTANT:
            # CUP BASE POINT
            # ==================================================

            pick_x = int(
                (x1 + x2) / 2
            )

            pick_y = int(y2)


            # ==================================================
            # LEFT / CENTRE / RIGHT
            # ==================================================

            if pick_x < 274:

                position = "LEFT"

            elif pick_x < 392:

                position = "CENTRE"

            else:

                position = "RIGHT"


            # ==================================================
            # WORKSPACE CHECK
            # ==================================================

            distance = cv2.pointPolygonTest(
                workspace_polygon,
                (
                    float(pick_x),
                    float(pick_y)
                ),
                True
            )


            inside_workspace = (
                distance >= -WORKSPACE_MARGIN
            )


            if inside_workspace:

                robot_x, robot_y = (
                    pixel_to_robot(
                        pick_x,
                        pick_y
                    )
                )


                recent_targets.append(
                    (
                        robot_x,
                        robot_y
                    )
                )


                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # Red = box centre
                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    5,
                    (0, 0, 255),
                    -1
                )


                # Purple = robot target point
                cv2.circle(
                    frame,
                    (
                        pick_x,
                        pick_y
                    ),
                    7,
                    (255, 0, 255),
                    -1
                )


                cv2.putText(
                    frame,
                    f"Cup - {position}",
                    (
                        x1,
                        max(y1 - 35, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    (
                        f"X={robot_x:.3f} "
                        f"Y={robot_y:.3f}"
                    ),
                    (
                        x1,
                        max(y1 - 10, 40)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


            else:

                recent_targets.append(
                    None
                )


        else:

            recent_targets.append(
                None
            )


        # ==================================================
        # CHECK STABILITY
        # ==================================================

        stable_target = (
            get_stable_target()
        )


        valid_count = sum(
            target is not None
            for target in recent_targets
        )


        if (
            stable_target is not None
            and
            not robot_has_moved
        ):

            raw_x, raw_y = stable_target

            # --------------------------------------------------
            # GRIPPER-CENTRE CORRECTION
            # --------------------------------------------------

            corrected_x, corrected_y, gripper_y_correction = (
                correct_gripper_centre(
                    raw_x,
                    raw_y
                )
            )


            # --------------------------------------------------
            # ROBOT REACH SAFETY CHECK
            #
            # IMPORTANT:
            # Safety is checked using the CORRECTED target,
            # because that is the position the robot will move to.
            # --------------------------------------------------

            corrected_reach = get_horizontal_reach(
                corrected_x,
                corrected_y
            )

            unsafe_x = (
                corrected_x > MAX_SAFE_X
            )

            unsafe_reach = (
                corrected_reach > MAX_SAFE_REACH
            )

            if unsafe_x or unsafe_reach:

                accepted_target = None

                cv2.putText(
                    frame,
                    "BEYOND SAFE ROBOT REACH",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    (
                        f"Corrected X={corrected_x:.3f}/{MAX_SAFE_X:.3f}"
                    ),
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Reach={corrected_reach:.3f}/{MAX_SAFE_REACH:.3f}"
                    ),
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Move cup closer to robot",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


            else:

                # The accepted target is now the corrected
                # gripper-centre target.
                accepted_target = (
                    corrected_x,
                    corrected_y
                )

                cv2.putText(
                    frame,
                    "TARGET STABLE",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    (
                        f"Camera X={raw_x:.3f} Y={raw_y:.3f}"
                    ),
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Gripper X={corrected_x:.3f} "
                        f"Y={corrected_y:.3f}"
                    ),
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Y correction = {gripper_y_correction*1000:.1f} mm"
                    ),
                    (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press M to MOVE",
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )


        elif not robot_has_moved:

            accepted_target = None

            cv2.putText(
                frame,
                "CHECKING TARGET...",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


        # Detection count
        cv2.putText(
            frame,
            f"Detections: {valid_count}/{WINDOW_SIZE}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        # ==================================================
        # DRAW WORKSPACE
        # ==================================================

        cv2.polylines(
            frame,
            [workspace_polygon],
            True,
            (255, 0, 0),
            2
        )


        cv2.line(
            frame,
            (274, 0),
            (274, 480),
            (0, 255, 255),
            1
        )


        cv2.line(
            frame,
            (392, 0),
            (392, 480),
            (0, 255, 255),
            1
        )


        cv2.imshow(
            "BRIO + YOLO + KINOVA",
            frame
        )


        # ==================================================
        # KEYBOARD
        # ==================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # Q = quit
        if key == ord("q"):

            print("User cancelled.")
            break


        # M = move robot
        if key == ord("m"):

            # Stable target exists, but it is outside the
            # experimentally validated robot X reach.
            if stable_target is not None:

                raw_check_x, raw_check_y = stable_target

                check_x, check_y, _ = (
                    correct_gripper_centre(
                        raw_check_x,
                        raw_check_y
                    )
                )

                check_reach = get_horizontal_reach(
                    check_x,
                    check_y
                )

                target_is_unsafe = (
                    check_x > MAX_SAFE_X
                    or
                    check_reach > MAX_SAFE_REACH
                )

            else:

                target_is_unsafe = False


            if target_is_unsafe:

                print()
                print("===================================")
                print("BEYOND SAFE ROBOT REACH")
                print("===================================")
                print(
                    f"Target X = {check_x:.3f} m"
                )
                print(
                    f"Maximum safe X = {MAX_SAFE_X:.3f} m"
                )
                print(
                    f"Target XY reach = {check_reach:.3f} m"
                )
                print(
                    f"Maximum safe XY reach = {MAX_SAFE_REACH:.3f} m"
                )
                print(
                    "Robot will NOT move."
                )
                print(
                    "Move the cup closer to the robot."
                )


            elif accepted_target is None:

                print()
                print(
                    "Target is NOT stable."
                )

                print(
                    "Robot will NOT move."
                )


            elif robot_has_moved:

                print(
                    "Robot already moved."
                )


            else:

                stable_x, stable_y = (
                    accepted_target
                )


                print()
                print("===================================")
                print("FINAL GRIPPER-CENTRE TARGET")
                print("===================================")

                print(
                    f"Corrected X = {stable_x:.3f}"
                )

                print(
                    f"Corrected Y = {stable_y:.3f}"
                )

                selected_reach = get_horizontal_reach(
                    stable_x,
                    stable_y
                )

                selected_theta_x = get_theta_x(
                    stable_x,
                    stable_y
                )

                print(
                    f"Z = {SAFE_Z:.3f}"
                )

                print(
                    f"XY Reach = {selected_reach:.3f} m"
                )

                print(
                    f"Theta X = {selected_theta_x:.1f} deg (reach-based)"
                )

                print()
                print(
                    "Moving robot ABOVE cup..."
                )


                # Close camera window before robot move
                # optional, but makes first test clearer
                cv2.destroyAllWindows()


                # success = move_above_cup(
                #     base,
                #     stable_x,
                #     stable_y
                # )
                # robot_has_moved = True

                success = True

                robot_has_moved = True

                if success:

                    print()
                    print("Robot is above corrected gripper-centre target.")

                    # --------------------------------------------------
                    # OPEN GRIPPER
                    # --------------------------------------------------

                    print()
                    print("Opening gripper...")

                    move_gripper(
                        base,
                        GRIPPER_OPEN
                    )

                    time.sleep(1.0)


                    # --------------------------------------------------
                    # ROTATE TO JSON PICK ORIENTATION
                    #
                    # Keep current corrected X/Y and SAFE_Z.
                    # Only orientation changes here.
                    # --------------------------------------------------

                    print()
                    print("======================================")
                    print("ROTATING TO JSON PICK ORIENTATION")
                    print("======================================")

                    print(
                        f"Theta X = {PICK_THETA_X:.3f}"
                    )
                    print(
                        f"Theta Y = {PICK_THETA_Y:.3f}"
                    )
                    print(
                        f"Theta Z = {PICK_THETA_Z:.3f}"
                    )

                    rotate_ok = move_to_pose(
                        base,
                        stable_x,
                        stable_y,
                        SAFE_Z,
                        PICK_THETA_X,
                        PICK_THETA_Y,
                        PICK_THETA_Z,
                        "Rotate To JSON Pick Orientation"
                    )


                    if not rotate_ok:

                        print()
                        print(
                            "Could not reach JSON pick orientation."
                        )
                        print(
                            "Robot will NOT descend."
                        )

                    else:

                        time.sleep(1.0)

                        # --------------------------------------------------
                        # JSON PICK ALIGNMENT CORRECTION
                        # --------------------------------------------------
                        # The normal camera/gripper-centre target is used
                        # for the hover. After rotating into the JSON pickup
                        # orientation, apply the current calibrated X/Y shift.
                        json_pick_x = (
                            stable_x
                            + JSON_PICK_X_OFFSET
                        )

                        json_pick_y = (
                            stable_y
                            + JSON_PICK_Y_OFFSET
                        )

                        json_pick_reach = get_horizontal_reach(
                            json_pick_x,
                            json_pick_y
                        )

                        print()
                        print("======================================")
                        print("JSON PICK ALIGNMENT CORRECTION")
                        print("======================================")
                        print(
                            f"Normal centred X = {stable_x:.3f} m"
                        )
                        print(
                            f"JSON X offset = {JSON_PICK_X_OFFSET:+.3f} m"
                        )
                        print(
                            f"Final JSON pick X = {json_pick_x:.3f} m"
                        )
                        print(
                            f"Final JSON pick Y = {json_pick_y:.3f} m"
                        )

                        # Safety check on the actual corrected pickup pose.
                        if (
                            json_pick_x > MAX_SAFE_X
                            or
                            json_pick_reach > MAX_SAFE_REACH
                        ):

                            print()
                            print(
                                "JSON pickup target is outside safe reach."
                            )
                            print(
                                "Robot will NOT align or descend."
                            )

                            align_ok = False

                        else:

                            print()
                            print(
                                "Applying JSON pickup alignment at SAFE_Z..."
                            )

                            align_ok = move_to_pose(
                                base,
                                json_pick_x,
                                json_pick_y,
                                SAFE_Z,
                                PICK_THETA_X,
                                PICK_THETA_Y,
                                PICK_THETA_Z,
                                "Align JSON Pick Position"
                            )


                        if not align_ok:

                            descend_ok = False

                            print()
                            print(
                                "JSON pickup alignment was not completed."
                            )

                        else:

                            time.sleep(1.0)

                            # --------------------------------------------------
                            # DESCEND USING CORRECTED JSON POSITION
                            # --------------------------------------------------

                            print()
                            print("======================================")
                            print("AUTOMATIC DESCENT")
                            print("======================================")

                            print(
                                f"Target pickup X = {json_pick_x:.3f} m"
                            )
                            print(
                                f"Target pickup Y = {json_pick_y:.3f} m"
                            )
                            print(
                                f"Target pickup Z = {PICK_Z:.3f} m"
                            )
                            print(
                                "Descending at corrected JSON position..."
                            )

                            descend_ok = move_to_pose(
                                base,
                                json_pick_x,
                                json_pick_y,
                                PICK_Z,
                                PICK_THETA_X,
                                PICK_THETA_Y,
                                PICK_THETA_Z,
                                "Descend To Pickup Height"
                            )


                        if descend_ok:

                            print()
                            print("======================================")
                            print("PICK HEIGHT REACHED")
                            print("======================================")

                            # ------------------------------------------
                            # CLOSE GRIPPER
                            # ------------------------------------------

                            print()
                            print("Closing gripper around cup...")

                            move_gripper(
                                base,
                                GRIPPER_CLOSE
                            )

                            print()
                            print(
                                f"Gripper closed to {GRIPPER_CLOSE:.2f}"
                            )

                            time.sleep(1.0)


                            # ------------------------------------------
                            # LIFT CUP WHILE KEEPING JSON ORIENTATION
                            # ------------------------------------------

                            print()
                            print(
                                "Lifting cup with JSON orientation..."
                            )

                            lift_ok = move_to_pose(
                                base,
                                json_pick_x,
                                json_pick_y,
                                SAFE_Z,
                                PICK_THETA_X,
                                PICK_THETA_Y,
                                PICK_THETA_Z,
                                "Lift Cup To Safe Height"
                            )


                            if lift_ok:

                                print()
                                print("======================================")
                                print("PICK AND LIFT COMPLETE")
                                print("======================================")

                                print(
                                    f"Cup lifted to Z = {SAFE_Z:.3f} m"
                                )
                                print(
                                    "JSON cup orientation maintained."
                                )
                                print(
                                    "Gripper remains CLOSED."
                                )
                                print(
                                    "Robot will STOP while holding the cup."
                                )

                                # ==========================================
                                # STEP 1: DETECT DESTINATION CUP
                                # ==========================================

                                print()
                                print(
                                    "Now looking for destination Cup 2..."
                                )

                                destination = (
                                    detect_destination_cup(
                                        cap,
                                        stable_x,
                                        stable_y
                                    )
                                )

                                if destination is not None:

                                    (
                                        destination_x,
                                        destination_y
                                    ) = destination

                                    print()
                                    print("======================================")
                                    print("STEP 1 SUCCESS")
                                    print("======================================")

                                    print(
                                        f"Cup 2 X = "
                                        f"{destination_x:.3f} m"
                                    )

                                    print(
                                        f"Cup 2 Y = "
                                        f"{destination_y:.3f} m"
                                    )

                                    print()
                                    print(
                                        "Cup 2 detection confirmed."
                                    )

                                    # ======================================
                                    # STEP 2: MOVE HELD CUP ABOVE CUP 2
                                    # ======================================

                                    move_to_destination_ok = (
                                        move_held_cup_above_destination(
                                            base,
                                            destination_x,
                                            destination_y
                                        )
                                    )

                                    if move_to_destination_ok:

                                        print()
                                        print(
                                            "Robot is above Cup 2 and upright."
                                        )

                                        print()
                                        print("======================================")
                                        print("STEP 6B: VOLUME-BASED PULSE-AND-WEIGH")
                                        print("======================================")
                                        print(
                                            "The robot now uses short pour pulses, "
                                            "returns upright, then weighs."
                                        )
                                        print()
                                        print(
                                            f"Pour begins near "
                                            f"{POUR_START_ACTUATOR_7:.2f} deg."
                                        )
                                        print(
                                            f"Maximum allowed pour angle = "
                                            f"{POUR_FULL_ACTUATOR_7:.2f} deg."
                                        )
                                        print()
                                        print(
                                            "Repeat the 25 g test so we can compare "
                                            "directly with the previous +25 g "
                                            "overshoot result."
                                        )
                                        print()

                                        scale_ser = (
                                            connect_kern_scale()
                                        )

                                        if scale_ser is None:

                                            print()
                                            print(
                                                "Scale connection failed."
                                            )
                                            print(
                                                "Robot will NOT pour."
                                            )

                                        else:

                                            try:

                                                tare_weight = (
                                                    capture_receiving_cup_tare(
                                                        scale_ser
                                                    )
                                                )

                                                (
                                                    liquid_key,
                                                    liquid_name,
                                                    liquid_density,
                                                    target_volume_ml,
                                                    target_weight
                                                ) = (
                                                    get_liquid_and_target_volume_from_user()
                                                )

                                                print()
                                                print(
                                                    f"Volume request accepted: "
                                                    f"{target_volume_ml:.2f} mL "
                                                    f"of {liquid_name}"
                                                )
                                                print(
                                                    f"KERN target mass = "
                                                    f"{target_weight:.2f} g"
                                                )

                                                confirm = (
                                                    input(
                                                        "Press P then Enter "
                                                        "to start volume-based "
                                                        "pouring, or anything "
                                                        "else to cancel: "
                                                    )
                                                    .strip()
                                                    .lower()
                                                )

                                                if confirm == "p":

                                                    pour_ok = (
                                                        target_weight_pour(
                                                            base,
                                                            scale_ser,
                                                            tare_weight,
                                                            target_weight,
                                                            liquid_name,
                                                            liquid_density,
                                                            target_volume_ml
                                                        )
                                                    )

                                                    if not pour_ok:

                                                        print()
                                                        print(
                                                            "Target-weight "
                                                            "pour did not "
                                                            "complete normally."
                                                        )

                                                    else:

                                                        print()
                                                        print(
                                                            "Pouring completed "
                                                            "successfully."
                                                        )
                                                        print(
                                                            "Starting automatic "
                                                            "cup return sequence..."
                                                        )

                                                        return_ok = (
                                                            return_cup_to_pickup_and_home(
                                                                base,
                                                                json_pick_x,
                                                                json_pick_y
                                                            )
                                                        )

                                                        if not return_ok:

                                                            print()
                                                            print(
                                                                "Final cup-return "
                                                                "sequence did not "
                                                                "complete normally."
                                                            )

                                                else:

                                                    print()
                                                    print(
                                                        "Target-weight pour "
                                                        "cancelled."
                                                    )

                                            finally:

                                                if scale_ser.is_open:

                                                    scale_ser.close()

                                                    print()
                                                    print(
                                                        "KERN scale "
                                                        "connection closed."
                                                    )

                                    else:

                                        print()
                                        print(
                                            "Step 2 movement was not completed."
                                        )
                                        print(
                                            "Robot remains holding Cup 1."
                                        )

                                else:

                                    print()
                                    print(
                                        "Cup 2 was not accepted."
                                    )

                            else:

                                print()
                                print(
                                    "Lift movement did not complete successfully."
                                )
                                print(
                                    "Gripper remains CLOSED."
                                )

                        else:

                            print()
                            print(
                                "Descent did not complete successfully."
                            )


                break


    cap.release()

    cv2.destroyAllWindows()


# ==========================================================
# 13. MAIN
# ==========================================================

def main():

    args = (
        utilities.parseConnectionArguments()
    )


    # ======================================================
    # CONNECT TO KINOVA
    # ======================================================

    with (
        utilities.DeviceConnection.createTcpConnection(
            args
        )
    ) as router:


        base = BaseClient(
            router
        )

        base_cyclic = BaseCyclicClient(
            router
        )


        print()
        print(
            "Connected to Kinova Gen3"
        )


        # ==================================================
        # SERVOING MODE
        # ==================================================

        servoing_mode = (
            Base_pb2.ServoingModeInformation()
        )

        servoing_mode.servoing_mode = (
            Base_pb2.SINGLE_LEVEL_SERVOING
        )

        base.SetServoingMode(
            servoing_mode
        )


        # ==================================================
        # HOME FIRST
        # ==================================================

        home_ok = move_home(
            base
        )


        if not home_ok:

            print()
            print(
                "Home failed."
            )

            print(
                "Stopping program."
            )

            return


        time.sleep(1)


        # ==================================================
        # START CAMERA DETECTION
        # ==================================================

        detect_cup_and_move(
            base
        )


if __name__ == "__main__":

    main()
