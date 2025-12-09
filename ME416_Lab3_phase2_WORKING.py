#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 14:05:32 2025

@author: adithichitiprolu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12-Robot Multi-Agent Path Planning Simulator
- Smaller robots (50% reduction)
- Spaced out goals (no overlap)
- Stationary robot avoidance (completed robots are obstacles)
- Deadlock reduction via:
    * Curvature-limited speed
    * Direction-aware deadlock detection
    * Reverse cooldown (no reverse/reverse loops)
    * Less frequent replanning, no replans close to goal
"""

import math
import numpy as np
import random
import time
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for multi-robot simulation"""
    
    # Arena bounds
    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 6.0
    ARENA_Y_MIN = 0.0
    ARENA_Y_MAX = 4.5
    
    # Robot parameters - REDUCED
    ROBOT_RADIUS = 0.15  # Reduced by 50% from 0.3
    GOAL_TOLERANCE = 0.15  # Adjusted proportionally
    
    # Control parameters
    LOOKAHEAD_DISTANCE = 0.2  # Set to 0.5m as specified
    MIN_LOOKAHEAD = 0.25      # Minimum lookahead = minimum turning radius
    V_DESIRED = 0.25
    MAX_ANGULAR_VEL = math.radians(60)  # 60 degrees per second
    MIN_TURNING_RADIUS = 0.25  # Minimum turning radius [m]
    dt = 0.05
    
    # A* Planning
    GRID_SIZE = 0.5
    
    # Multi-robot coordination
    BASE_SAFETY_RADIUS = 0.4           # Slightly smaller to be less paranoid
    VELOCITY_SCALE_FACTOR = 0.2
    PREDICTION_HORIZON = 1.5           # Modest prediction horizon
    REPLAN_INTERVAL = 0.7              # Slower replanning to avoid thrashing
    
    # Proximity / deadlock thresholds
    CRITICAL_OBSTACLE_DISTANCE = 0.05  # Reverse only when REALLY close
    GOAL_NO_REPLAN_RADIUS = 0.7        # Do not replan when closer than this to goal
    REVERSE_COOLDOWN = 2.0             # Seconds after reversing during which we don't reverse again


# =============================================================================
# UTILITIES
# =============================================================================

def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def distance(x1, y1, x2, y2):
    """Euclidean distance"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# =============================================================================
# A* PLANNER
# =============================================================================

class AStarPlanner:
    """A* grid-based path planner"""
    
    def __init__(self, obstacles, resolution, rr, bounds):
        """
        Initialize grid map for A* planning
        
        obstacles: list of (x, y) obstacle positions
        resolution: grid resolution [m]
        rr: robot radius [m]
        bounds: (min_x, max_x, min_y, max_y)
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.motion = self.get_motion_model()
        self.obstacle_map = self.calc_obstacle_map(obstacles)
    
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index
    
    def planning(self, sx, sy, gx, gy):
        """A* path search"""
        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0, -1
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0, -1
        )
        
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        
        while True:
            if len(open_set) == 0:
                return None, None
            
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])
            )
            current = open_set[c_id]
            
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                return self.calc_final_path(goal_node, closed_set)
            
            del open_set[c_id]
            closed_set[c_id] = current
            
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id
                )
                n_id = self.calc_grid_index(node)
                
                if not self.verify_node(node):
                    continue
                
                if n_id in closed_set:
                    continue
                
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
    
    def calc_final_path(self, goal_node, closed_set):
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d
    
    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos
    
    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)
    
    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)
    
    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        
        if px < self.min_x or py < self.min_y:
            return False
        if px >= self.max_x or py >= self.max_y:
            return False
        
        if self.obstacle_map[node.x][node.y]:
            return False
        
        return True
    
    def calc_obstacle_map(self, obstacles):
        obstacle_map = [[False for _ in range(self.y_width)]
                       for _ in range(self.x_width)]
        
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                
                # Check boundaries
                if x < self.min_x + 0.3 or x > self.max_x - 0.3:
                    obstacle_map[ix][iy] = True
                    continue
                if y < self.min_y + 0.3 or y > self.max_y - 0.3:
                    obstacle_map[ix][iy] = True
                    continue
                
                # Check obstacles
                for ox, oy in obstacles:
                    d = math.hypot(ox - x, oy - y)
                    if d <= self.rr:
                        obstacle_map[ix][iy] = True
                        break
        
        return obstacle_map
    
    @staticmethod
    def get_motion_model():
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)]
        ]
        return motion


# =============================================================================
# ROBOT CLASS
# =============================================================================

class Robot:
    """Single robot with path planning and control"""
    
    def __init__(self, robot_id, x, y, goal_x, goal_y, color, cfg):
        self.id = robot_id
        self.x = x
        self.y = y
        self.theta = random.uniform(0, 2 * math.pi)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.color = color
        self.cfg = cfg
        
        # State
        self.vx = 0.0
        self.vy = 0.0
        self.completed = False
        self.last_replan_time = 0.0
        
        # Stationary detection
        self.is_stationary = False
        self.stationary_start_time = None
        self.last_speed = 0.0
        self.stationary_threshold = 0.01      # Speed below this is considered stopped [m/s]
        self.stationary_time_threshold = 5.0  # Time stopped before considered stationary [s]
        self.max_stationary_time = 15.0       # Force reversing after this time if still stuck [s]
        
        # Reversing for deadlock recovery
        self.is_reversing = False
        self.reverse_start_time = None
        self.reverse_duration = 1.0           # Reverse for 1 second
        self.reverse_distance_target = 0.3    # Try to reverse 30cm
        self.reverse_start_position = None
        self.last_reverse_end_time = -1e9     # For reverse cooldown
        
        # Path attempt tracking to avoid repeating failed paths
        self.attempted_paths = []  # List of path signatures we've tried
        self.max_path_history = 5  # Remember last 5 path attempts
        self.path_attempt_count = 0  # Number of times we've tried planning
        self.last_successful_progress_time = 0.0  # Last time we made progress
        self.progress_threshold = 0.2  # Must move 20cm to count as progress
        
        # Path
        self.path_x = [x, goal_x]
        self.path_y = [y, goal_y]
        
        # Trajectory
        self.trajectory_x = [x]
        self.trajectory_y = [y]
    
    # -------------------------------------------------------------------------
    # Path bookkeeping
    # -------------------------------------------------------------------------
    
    def get_path_signature(self):
        """
        Create a signature for the current path to detect if we're trying the same path again.
        Uses first few waypoints as signature.
        """
        if len(self.path_x) < 2:
            return None
        
        # Use first 3 waypoints (or all if fewer) as signature
        num_points = min(3, len(self.path_x))
        signature = []
        for i in range(num_points):
            # Round to 0.2m grid to allow for small variations
            sig_x = round(self.path_x[i] / 0.2) * 0.2
            sig_y = round(self.path_y[i] / 0.2) * 0.2
            signature.append((sig_x, sig_y))
        
        return tuple(signature)
    
    def is_path_already_attempted(self, new_path_x, new_path_y):
        """Check if this path (or very similar) has been tried before"""
        if len(new_path_x) < 2:
            return False
        
        # Create signature for new path
        num_points = min(3, len(new_path_x))
        new_signature = []
        for i in range(num_points):
            sig_x = round(new_path_x[i] / 0.2) * 0.2
            sig_y = round(new_path_y[i] / 0.2) * 0.2
            new_signature.append((sig_x, sig_y))
        new_signature = tuple(new_signature)
        
        # Check against attempted paths
        return new_signature in self.attempted_paths
    
    def record_path_attempt(self):
        """Record current path as attempted"""
        signature = self.get_path_signature()
        if signature and signature not in self.attempted_paths:
            self.attempted_paths.append(signature)
            # Keep only recent attempts
            if len(self.attempted_paths) > self.max_path_history:
                self.attempted_paths.pop(0)
    
    def calculate_detour_waypoint(self, attempt_number, other_robots):
        """
        Calculate a detour waypoint based on attempt number.
        Each attempt uses a different strategy to find a new path.
        """
        angle_to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        
        # Strategy varies by attempt number
        strategy = attempt_number % 4
        
        if strategy == 0:
            # Try going right
            detour_angle = angle_to_goal + math.pi / 2.5
            detour_distance = 1.5
        elif strategy == 1:
            # Try going left
            detour_angle = angle_to_goal - math.pi / 2.5
            detour_distance = 1.5
        elif strategy == 2:
            # Try sharp right
            detour_angle = angle_to_goal + math.pi / 2
            detour_distance = 1.8
        else:
            # Try sharp left
            detour_angle = angle_to_goal - math.pi / 2
            detour_distance = 1.8
        
        # Calculate waypoint
        wp_x = self.x + detour_distance * math.cos(detour_angle)
        wp_y = self.y + detour_distance * math.sin(detour_angle)
        
        # Make sure it's not near other stationary robots
        min_clearance = 1.2
        for other in other_robots:
            if other.id == self.id:
                continue
            if other.completed or other.is_stationary:
                dist = distance(wp_x, wp_y, other.x, other.y)
                if dist < min_clearance:
                    # Too close, push waypoint away
                    push_angle = math.atan2(wp_y - other.y, wp_x - other.x)
                    wp_x = other.x + min_clearance * math.cos(push_angle)
                    wp_y = other.y + min_clearance * math.sin(push_angle)
        
        # Clamp to bounds
        margin = 0.5
        wp_x = np.clip(wp_x, self.cfg.ARENA_X_MIN + margin, self.cfg.ARENA_X_MAX - margin)
        wp_y = np.clip(wp_y, self.cfg.ARENA_Y_MIN + margin, self.cfg.ARENA_Y_MAX - margin)
        
        return wp_x, wp_y
    
    def is_path_blocked(self, other_robots, lookahead_points=3):
        """
        Check if the current path has robots blocking it.
        Returns True if any robot is too close to upcoming waypoints.
        Only checks very near waypoints to avoid false positives.
        """
        if len(self.path_x) < 2:
            return False
        
        # Only check next few waypoints
        num_points_to_check = min(lookahead_points, len(self.path_x))
        
        for i in range(num_points_to_check):
            waypoint_x = self.path_x[i]
            waypoint_y = self.path_y[i]
            
            # Check if any robot (moving or stationary) is blocking this waypoint
            for other in other_robots:
                if other.id == self.id:
                    continue
                
                dist_to_waypoint = distance(waypoint_x, waypoint_y, other.x, other.y)
                
                # Only consider blocked if VERY close
                blocking_radius = self.cfg.BASE_SAFETY_RADIUS * 0.8
                if other.completed or other.is_stationary:
                    blocking_radius = self.cfg.BASE_SAFETY_RADIUS * 1.2
                
                if dist_to_waypoint < blocking_radius:
                    return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Deadlock / reversing logic
    # -------------------------------------------------------------------------
    
    def check_for_deadlock(self, current_time, other_robots):
        """
        Check if robot should reverse.
        Conditions:
        1. Within critical distance in FRONT of any obstacle (immediate reverse)
        2. Stationary for >max_stationary_time (deadlock reverse)
        Reverse cooldown prevents repeated reverse/reverse loops.
        """
        if self.is_reversing:
            return False  # Already reversing
        
        # Cooldown after finishing a reverse to prevent oscillation
        if current_time - self.last_reverse_end_time < self.cfg.REVERSE_COOLDOWN:
            return False
        
        # Check proximity to ALL obstacles (immediate reverse condition)
        for other in other_robots:
            if other.id == self.id:
                continue
            
            dist = distance(self.x, self.y, other.x, other.y)
            
            # Only care if the obstacle is roughly in front of us
            bearing_to_other = math.atan2(other.y - self.y, other.x - self.x)
            heading_error = abs(wrap_to_pi(bearing_to_other - self.theta))
            
            # "In front" = within 90 degrees
            in_front = heading_error < (math.pi / 2.0)
            
            critical = self.cfg.CRITICAL_OBSTACLE_DISTANCE + self.cfg.ROBOT_RADIUS * 1.2
            if dist < critical and in_front:
                print(f"  Robot {self.id + 1}: CRITICAL! {dist:.3f}m to obstacle ahead - reversing")
                return True
        
        # Check stationary timeout (deadlock condition)
        if self.stationary_start_time is not None:
            stopped_duration = current_time - self.stationary_start_time
            if stopped_duration >= self.max_stationary_time:
                print(f"  Robot {self.id + 1}: Stationary for {stopped_duration:.1f}s - reversing")
                return True
        
        return False
    
    def start_reversing(self, current_time):
        """Initiate reversing maneuver to escape deadlock"""
        if not self.is_reversing:
            self.is_reversing = True
            self.reverse_start_time = current_time
            self.reverse_start_position = (self.x, self.y)
            print(f"  Robot {self.id + 1}: DEADLOCK! Reversing to escape...")
    
    def update_reversing(self, current_time):
        """
        Check if reversing maneuver is complete.
        Returns (is_still_reversing, v_reverse, omega_reverse)
        """
        if not self.is_reversing:
            return False, 0.0, 0.0
        
        # Check if we've reversed long enough
        reverse_time = current_time - self.reverse_start_time
        
        # Check distance reversed
        if self.reverse_start_position:
            dist_reversed = distance(self.x, self.y, 
                                    self.reverse_start_position[0], 
                                    self.reverse_start_position[1])
        else:
            dist_reversed = 0
        
        # Stop reversing if: reversed long enough OR traveled target distance
        if reverse_time >= self.reverse_duration or dist_reversed >= self.reverse_distance_target:
            print(f"  Robot {self.id + 1}: Finished reversing ({dist_reversed:.2f}m)")
            self.is_reversing = False
            self.reverse_start_time = None
            self.reverse_start_position = None
            # Reset stationary timer
            self.stationary_start_time = None
            self.is_stationary = False
            # Increment attempt count to force new strategy next time
            self.path_attempt_count += 1
            # Clear recent path history to allow more options
            if len(self.attempted_paths) > 2:
                self.attempted_paths = self.attempted_paths[-2:]
            # Mark time for reverse cooldown
            self.last_reverse_end_time = current_time
            return False, 0.0, 0.0
        
        # Continue reversing - move backward slowly
        v_reverse = -0.15  # Reverse at 15 cm/s
        omega_reverse = 0.0  # Go straight back
        
        return True, v_reverse, omega_reverse
    
    def update_stationary_status(self, current_time):
        """
        Update whether robot is considered stationary BY OTHER ROBOTS.
        A robot is stationary if it has been stopped for more than 5 seconds.
        IMPORTANT: The robot itself doesn't stop moving - it just gets flagged
        so other robots can avoid it better.
        """
        current_speed = math.hypot(self.vx, self.vy)
        
        # Check if robot is stopped
        if current_speed < self.stationary_threshold:
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
            else:
                # Check how long it's been stopped
                stopped_duration = current_time - self.stationary_start_time
                if stopped_duration >= self.stationary_time_threshold:
                    if not self.is_stationary:
                        self.is_stationary = True
                        print(f"  Robot {self.id + 1}: Marked as STATIONARY (stopped {stopped_duration:.1f}s)")
        else:
            # Robot is moving - reset stationary status
            if self.is_stationary:
                print(f"  Robot {self.id + 1}: Resuming - no longer stationary")
            self.is_stationary = False
            self.stationary_start_time = None
        
        self.last_speed = current_speed
    
    # -------------------------------------------------------------------------
    # Local collision-aware speed scaling
    # -------------------------------------------------------------------------
    
    def check_nearby_robots(self, other_robots):
        """
        Check for nearby robots and adjust velocity accordingly.
        Symmetric for all robots (same code, same rules).
        """
        min_distance = float('inf')
        
        for other in other_robots:
            if other.id == self.id:
                continue
            
            dist = distance(self.x, self.y, other.x, other.y)
            min_distance = min(min_distance, dist)
        
        # Default: full speed
        velocity_scale = 1.0
        
        # Critical distance - emergency slow
        critical_distance = (self.cfg.ROBOT_RADIUS * 2) + 0.15
        if min_distance < critical_distance:
            velocity_scale = 0.05  # Very slow
        else:
            # Collision zone - aggressive slowdown
            collision_zone = 0.8
            if min_distance < collision_zone:
                velocity_scale = (min_distance - critical_distance) / (collision_zone - critical_distance)
                velocity_scale = max(0.15, velocity_scale ** 2)
        
        # Clamp to [0.05, 1.0]
        velocity_scale = max(0.05, min(1.0, velocity_scale))
        
        return velocity_scale
    
    # -------------------------------------------------------------------------
    # Pure pursuit controller
    # -------------------------------------------------------------------------
    
    def pure_pursuit_control(self, other_robots, current_time):
        """Pure Pursuit controller with collision avoidance for ALL robots"""
        if self.completed:
            return 0.0, 0.0
        
        # Check if we're in reversing mode
        is_reversing, v_reverse, omega_reverse = self.update_reversing(current_time)
        if is_reversing:
            return v_reverse, omega_reverse
        
        dist_to_goal = distance(self.x, self.y, self.goal_x, self.goal_y)
        
        if dist_to_goal < self.cfg.GOAL_TOLERANCE:
            self.completed = True
            return 0.0, 0.0
        
        # Check proximity to ALL robots (moving and stationary)
        velocity_scale = self.check_nearby_robots(other_robots)
        
        # Final approach mode
        if dist_to_goal < 0.4:
            alpha = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
            heading_error = wrap_to_pi(alpha - self.theta)
            v_scale = min(1.0, dist_to_goal / 0.3) * velocity_scale
            v = self.cfg.V_DESIRED * v_scale
            omega = np.clip(2.0 * heading_error, 
                            -self.cfg.MAX_ANGULAR_VEL, 
                            self.cfg.MAX_ANGULAR_VEL)
            return v, omega
        
        # Find lookahead point
        distances = [distance(self.x, self.y, px, py) 
                     for px, py in zip(self.path_x, self.path_y)]
        closest_idx = np.argmin(distances)
        
        # Calculate arc length along path
        path_s = [0.0]
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i-1]
            dy = self.path_y[i] - self.path_y[i-1]
            path_s.append(path_s[-1] + math.hypot(dx, dy))
        
        s_closest = path_s[closest_idx]
        
        # Adaptive lookahead based on velocity scale
        adaptive_lookahead = self.cfg.LOOKAHEAD_DISTANCE * velocity_scale
        adaptive_lookahead = max(0.3, adaptive_lookahead)  # Minimum lookahead
        
        s_lookahead = s_closest + adaptive_lookahead
        
        if s_lookahead > path_s[-1]:
            lookahead_idx = len(self.path_x) - 1
        else:
            lookahead_idx = closest_idx
            for i in range(closest_idx, len(path_s)):
                if path_s[i] >= s_lookahead:
                    lookahead_idx = i
                    break
        
        lookahead_x = self.path_x[lookahead_idx]
        lookahead_y = self.path_y[lookahead_idx]
        
        alpha = math.atan2(lookahead_y - self.y, lookahead_x - self.x)
        
        # Calculate actual lookahead distance and enforce minimum
        actual_ld = distance(self.x, self.y, lookahead_x, lookahead_y)
        actual_ld = max(actual_ld, self.cfg.MIN_TURNING_RADIUS)  # Enforce minimum
        
        curvature = 2 * math.sin(wrap_to_pi(alpha - self.theta)) / actual_ld
        
        # Apply velocity scaling but also respect turning limits:
        # |omega| = |curvature * v| <= MAX_ANGULAR_VEL
        base_v = self.cfg.V_DESIRED * velocity_scale

        if abs(curvature) > 1e-3:
            v_turn_limited = self.cfg.MAX_ANGULAR_VEL / abs(curvature)
            v = min(base_v, v_turn_limited)
        else:
            v = base_v

        # Avoid coming to a complete stop unless we explicitly want it
        v = max(0.05, v)

        omega = curvature * v
        omega = np.clip(omega, -self.cfg.MAX_ANGULAR_VEL, self.cfg.MAX_ANGULAR_VEL)
        
        return v, omega
    
    # -------------------------------------------------------------------------
    # Replanning
    # -------------------------------------------------------------------------
    
    def replan(self, other_robots, current_time):
        """Replan path considering other robots as dynamic obstacles"""
        if self.completed:
            return
        
        # Don't replan while reversing - let it complete
        if self.is_reversing:
            return

        # Avoid thrashing near the goal: only intervene if truly stuck/critical
        dist_to_goal = distance(self.x, self.y, self.goal_x, self.goal_y)
        if dist_to_goal < self.cfg.GOAL_NO_REPLAN_RADIUS:
            if self.check_for_deadlock(current_time, other_robots):
                self.start_reversing(current_time)
            return
        
        # Check for critical proximity or deadlock - initiate reversing if needed
        should_reverse = self.check_for_deadlock(current_time, other_robots)
        if should_reverse:
            self.start_reversing(current_time)
            return  # Will replan after reversing completes
        
        # Check if current path is blocked - if so, force immediate replan
        path_blocked = self.is_path_blocked(other_robots)
        
        if not path_blocked and current_time - self.last_replan_time < self.cfg.REPLAN_INTERVAL:
            return
        
        if path_blocked:
            print(f"  Robot {self.id + 1}: Path blocked! Immediate replan (attempt #{self.path_attempt_count})")
        
        # Collect dynamic obstacles from ALL other robots (including completed ones)
        obstacles = []
        for other in other_robots:
            if other.id == self.id:
                continue
            
            # Treat completed OR stationary robots as static obstacles
            if other.completed or other.is_stationary:
                radius = self.cfg.BASE_SAFETY_RADIUS * 1.6  # Big but not insane
                # Add rings around stationary robot
                for k in range(12):
                    angle = 2 * math.pi * k / 12
                    ox = other.x + radius * math.cos(angle)
                    oy = other.y + radius * math.sin(angle)
                    obstacles.append((ox, oy))
                for k in range(8):
                    angle = 2 * math.pi * k / 8
                    ox = other.x + radius * 0.6 * math.cos(angle)
                    oy = other.y + radius * 0.6 * math.sin(angle)
                    obstacles.append((ox, oy))
            else:
                # Moving robots - use predictive obstacles
                speed = math.hypot(other.vx, other.vy)
                pred_x = other.x + other.vx * self.cfg.PREDICTION_HORIZON
                pred_y = other.y + other.vy * self.cfg.PREDICTION_HORIZON
                radius = self.cfg.BASE_SAFETY_RADIUS + speed * self.cfg.VELOCITY_SCALE_FACTOR
                
                # Add points around predicted position
                for k in range(8):
                    angle = 2 * math.pi * k / 8
                    ox = pred_x + radius * math.cos(angle)
                    oy = pred_y + radius * math.sin(angle)
                    obstacles.append((ox, oy))
        
        # Plan new path
        bounds = (self.cfg.ARENA_X_MIN, self.cfg.ARENA_X_MAX,
                  self.cfg.ARENA_Y_MIN, self.cfg.ARENA_Y_MAX)
        
        # Use detour waypoint only if we've failed multiple times
        goal_x, goal_y = self.goal_x, self.goal_y
        use_detour = self.path_attempt_count >= 2  # Only after 2+ failures
        
        if use_detour:
            goal_x, goal_y = self.calculate_detour_waypoint(self.path_attempt_count, other_robots)
            print(f"    → Using detour strategy #{self.path_attempt_count % 4} to waypoint ({goal_x:.2f}, {goal_y:.2f})")
        
        max_planning_attempts = 3
        for planning_attempt in range(max_planning_attempts):
            try:
                planner = AStarPlanner(obstacles, self.cfg.GRID_SIZE, 
                                       self.cfg.ROBOT_RADIUS, bounds)
                rx, ry = planner.planning(self.x, self.y, goal_x, goal_y)
                
                if rx is not None and len(rx) > 1:
                    # Check if this path is different from previous attempts
                    if not self.is_path_already_attempted(rx[::-1], ry[::-1]):
                        self.path_x = rx[::-1]
                        self.path_y = ry[::-1]
                        self.record_path_attempt()
                        self.last_replan_time = current_time
                        print(f"    ✓ Found NEW path with {len(self.path_x)} waypoints")
                        return
                    else:
                        # This path was tried before, try different detour
                        print(f"    ⚠ Path already attempted, trying different strategy...")
                        self.path_attempt_count += 1
                        goal_x, goal_y = self.calculate_detour_waypoint(self.path_attempt_count, other_robots)
                        continue
                else:
                    # Planning failed, try different detour
                    self.path_attempt_count += 1
                    goal_x, goal_y = self.calculate_detour_waypoint(self.path_attempt_count, other_robots)
                    
            except Exception as e:
                print(f"    ⚠ Planning error: {e}")
                self.path_attempt_count += 1
                goal_x, goal_y = self.calculate_detour_waypoint(self.path_attempt_count, other_robots)
        
        # If all planning attempts failed, keep current path
        print(f"    ⚠ All planning attempts failed, keeping current path")
        self.last_replan_time = current_time
    
    # -------------------------------------------------------------------------
    # State update
    # -------------------------------------------------------------------------
    
    def update(self, other_robots, current_time):
        """Update robot state"""
        if self.completed:
            return
        
        # Track progress - reset attempt counter if making good progress
        if not hasattr(self, 'last_progress_position'):
            self.last_progress_position = (self.x, self.y)
            self.last_progress_time = current_time
        
        progress_distance = distance(self.x, self.y, 
                                     self.last_progress_position[0], 
                                     self.last_progress_position[1])
        
        if progress_distance > 0.5:  # Made 50cm of progress
            if self.path_attempt_count > 0:
                print(f"  Robot {self.id + 1}: Made progress, resetting attempt counter")
            self.path_attempt_count = 0
            self.last_progress_position = (self.x, self.y)
            self.last_progress_time = current_time
        
        # Update stationary status
        self.update_stationary_status(current_time)
        
        # Replan if needed
        self.replan(other_robots, current_time)
        
        # Get control commands
        v, omega = self.pure_pursuit_control(other_robots, current_time)
        
        # Update state
        self.theta += omega * self.cfg.dt
        self.theta = wrap_to_pi(self.theta)
        
        new_x = self.x + v * math.cos(self.theta) * self.cfg.dt
        new_y = self.y + v * math.sin(self.theta) * self.cfg.dt
        
        self.vx = (new_x - self.x) / self.cfg.dt
        self.vy = (new_y - self.y) / self.cfg.dt
        
        self.x = new_x
        self.y = new_y
        
        self.trajectory_x.append(self.x)
        self.trajectory_y.append(self.y)


# =============================================================================
# SIMULATION
# =============================================================================

def initialize_robots(cfg, num_robots=12):
    """Initialize robots with random start and goal positions"""
    robots = []
    margin = 0.5
    
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
        '#E63946', '#A8DADC'
    ]
    
    # Keep track of all goal positions to ensure no overlap
    all_goals = []
    
    def is_valid_position(x, y, existing_robots, is_goal=False):
        if (x < cfg.ARENA_X_MIN + margin or x > cfg.ARENA_X_MAX - margin or
            y < cfg.ARENA_Y_MIN + margin or y > cfg.ARENA_Y_MAX - margin):
            return False
        
        # Check against existing robot start positions
        for robot in existing_robots:
            if distance(x, y, robot.x, robot.y) < 1.5:  # Increased spacing
                return False
        
        # Check against existing goal positions
        for gx, gy in all_goals:
            if distance(x, y, gx, gy) < 1.5:  # Increased spacing
                return False
        
        # If this is a goal, also check it doesn't overlap with other goals
        if is_goal:
            for robot in existing_robots:
                if distance(x, y, robot.goal_x, robot.goal_y) < 1.5:
                    return False
        
        return True
    
    for i in range(num_robots):
        # Find valid start position
        attempts = 0
        while attempts < 100:
            x = random.uniform(cfg.ARENA_X_MIN + margin, cfg.ARENA_X_MAX - margin)
            y = random.uniform(cfg.ARENA_Y_MIN + margin, cfg.ARENA_Y_MAX - margin)
            if is_valid_position(x, y, robots, is_goal=False):
                break
            attempts += 1
        
        # Find valid goal position (ensuring no goal overlap)
        attempts = 0
        while attempts < 100:
            goal_x = random.uniform(cfg.ARENA_X_MIN + margin, cfg.ARENA_X_MAX - margin)
            goal_y = random.uniform(cfg.ARENA_Y_MIN + margin, cfg.ARENA_Y_MAX - margin)
            
            if is_valid_position(goal_x, goal_y, robots, is_goal=True):
                goal_valid = True
                for gx, gy in all_goals:
                    if distance(goal_x, goal_y, gx, gy) < 1.5:
                        goal_valid = False
                        break
                
                if goal_valid:
                    all_goals.append((goal_x, goal_y))
                    break
            attempts += 1
        
        robot = Robot(i, x, y, goal_x, goal_y, colors[i], cfg)
        robots.append(robot)
    
    return robots


def run_simulation(animate=True):
    """Run the multi-robot simulation"""
    cfg = Config()
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("12-ROBOT MULTI-AGENT PATH PLANNING SIMULATOR")
    print("Features:")
    print("  • Lookahead: 0.5m, Replan every: {:.1f}s".format(cfg.REPLAN_INTERVAL))
    print("  • Reverse if within {:.2f}m of obstacle in front".format(
        cfg.CRITICAL_OBSTACLE_DISTANCE + cfg.ROBOT_RADIUS * 1.2))
    print("  • Reverse if stopped >{:.0f}s".format(15.0))
    print("  • Reverse cooldown: {:.1f}s".format(cfg.REVERSE_COOLDOWN))
    print("=" * 60)
    print("Initializing robots...")
    
    robots = initialize_robots(cfg, num_robots=12)
    print(f"✓ Initialized {len(robots)} robots")
    print(f"✓ Robot radius: {cfg.ROBOT_RADIUS}m")
    print(f"✓ All goals spaced >1.5m apart")
    
    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(cfg.ARENA_X_MIN - 0.5, cfg.ARENA_X_MAX + 0.5)
        ax.set_ylim(cfg.ARENA_Y_MIN - 0.5, cfg.ARENA_Y_MAX + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('12-Robot Navigation (Stationary Robots = Obstacles)')
        
        # Plot elements
        robot_plots = []
        path_plots = []
        traj_plots = []
        heading_arrows = []
        
        for robot in robots:
            # Goal (matched to robot radius)
            goal_circle = Circle((robot.goal_x, robot.goal_y), cfg.ROBOT_RADIUS,
                                 color=robot.color, alpha=0.3, zorder=1)
            ax.add_patch(goal_circle)
            goal_edge = Circle((robot.goal_x, robot.goal_y), cfg.ROBOT_RADIUS,
                               fill=False, edgecolor=robot.color, linewidth=2, zorder=1)
            ax.add_patch(goal_edge)
            
            # Path
            path_line, = ax.plot([], [], color=robot.color, alpha=0.3, 
                                 linewidth=1, zorder=2)
            path_plots.append(path_line)
            
            # Trajectory
            traj_line, = ax.plot([], [], color=robot.color, alpha=0.7,
                                 linewidth=2, zorder=3)
            traj_plots.append(traj_line)
            
            # Robot body
            robot_circle = Circle((robot.x, robot.y), cfg.ROBOT_RADIUS,
                                  color=robot.color, zorder=4)
            ax.add_patch(robot_circle)
            robot_edge = Circle((robot.x, robot.y), cfg.ROBOT_RADIUS,
                                fill=False, edgecolor='white', linewidth=2, zorder=4)
            ax.add_patch(robot_edge)
            
            # Robot ID
            robot_text = ax.text(robot.x, robot.y, str(robot.id + 1),
                                 ha='center', va='center', color='white',
                                 fontsize=8, fontweight='bold', zorder=5)
            robot_plots.append((robot_circle, robot_edge, robot_text))
            
            # Heading arrow
            arrow = ax.arrow(robot.x, robot.y, 0, 0, head_width=0.08,
                             head_length=0.08, fc='white', ec='white', zorder=5)
            heading_arrows.append(arrow)
        
        plt.tight_layout()
    
    # Main simulation loop
    current_time = 0.0
    iteration = 0
    print("\nStarting simulation...")
    print("Behavior:")
    print("  • Near obstacle: Slow down & replan")
    print("  • Critical close in front: REVERSE (Red)")
    print("  • Stopped >5s: Orange (stationary)")
    print("  • Stopped >15s: REVERSE (Red)")
    print("  • Completed: Green\n")
    
    try:
        while True:
            # Update all robots
            for robot in robots:
                robot.update(robots, current_time)
            
            # Check completion
            completed = sum(1 for r in robots if r.completed)
            if completed == len(robots):
                print(f"\n✓ All robots reached their goals!")
                print(f"  Total time: {current_time:.1f}s")
                break
            
            # Update visualization
            if animate and iteration % 5 == 0:
                for i, robot in enumerate(robots):
                    robot_circle, robot_edge, robot_text = robot_plots[i]
                    robot_circle.center = (robot.x, robot.y)
                    robot_edge.center = (robot.x, robot.y)
                    robot_text.set_position((robot.x, robot.y))
                    
                    # Change color based on status
                    if robot.completed:
                        robot_circle.set_color('#2ecc71')  # Green for completed
                        robot_edge.set_linewidth(3)
                    elif robot.is_reversing:
                        robot_circle.set_color('#FF0000')  # Red for reversing
                        robot_edge.set_linewidth(3)
                    elif robot.is_stationary:
                        robot_circle.set_color('#FFA500')  # Orange for stationary
                        robot_edge.set_linewidth(3)
                    else:
                        robot_circle.set_color(robot.color)
                        robot_edge.set_linewidth(2)
                    
                    # Update path and trajectory
                    path_plots[i].set_data(robot.path_x, robot.path_y)
                    traj_plots[i].set_data(robot.trajectory_x, robot.trajectory_y)
                    
                    # Update heading arrow
                    if not robot.completed and heading_arrows[i] in ax.patches:
                        heading_arrows[i].remove()
                    if not robot.completed:
                        dx = 0.3 * math.cos(robot.theta)
                        dy = 0.3 * math.sin(robot.theta)
                        heading_arrows[i] = ax.arrow(
                            robot.x, robot.y, dx, dy,
                            head_width=0.08, head_length=0.08,
                            fc='white', ec='white', zorder=5
                        )
                
                reversing_count = sum(1 for r in robots if r.is_reversing)
                stationary_count = sum(
                    1 for r in robots
                    if r.is_stationary and not r.completed and not r.is_reversing
                )
                ax.set_title(
                    f'Time: {current_time:.1f}s | '
                    f'Completed: {completed}/{len(robots)} | '
                    f'Reversing: {reversing_count} (Red) | '
                    f'Stationary: {stationary_count} (Orange)'
                )
                plt.pause(0.001)
            
            # Print progress periodically
            if iteration % 100 == 0:
                reversing_count = sum(1 for r in robots if r.is_reversing)
                stationary_count = sum(
                    1 for r in robots
                    if r.is_stationary and not r.completed and not r.is_reversing
                )
                print(
                    f"t={current_time:.1f}s | "
                    f"Completed: {completed}/{len(robots)} | "
                    f"Reversing: {reversing_count} | "
                    f"Stationary: {stationary_count}"
                )
            
            current_time += cfg.dt
            iteration += 1
            time.sleep(cfg.dt * 0.5)  # Slow down for visualization
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    if animate:
        print("\nClose the plot window to exit.")
        plt.ioff()
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n12-Robot Multi-Agent Path Planning Simulator")
    print("Smaller robots + Stationary obstacle avoidance + Deadlock reduction\n")
    run_simulation(animate=True)
