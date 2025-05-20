import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.patches import Circle
import heapq

@dataclass
class Node:
    id: int
    x: float
    y: float
    is_priority: bool
    status: str  # 'waiting', 'picked', 'dropped'

class Vehicle:
    def __init__(self, speed=35):
        self.speed = speed  # km/h
        self.current_position = (0, 0)
        self.passengers = []
        self.total_distance = 0
        self.time_elapsed = 0
        self.path_history = [(0, 0)]  # Track path for visualization

    def move_to(self, destination: Tuple[float, float]):
        distance = np.sqrt(
            (destination[0] - self.current_position[0])**2 + 
            (destination[1] - self.current_position[1])**2
        )
        time_taken = distance / self.speed
        self.current_position = destination
        self.total_distance += distance
        self.time_elapsed += time_taken
        self.path_history.append(destination)  # Record the movement
        return distance, time_taken

class RoutingSystem:
    def __init__(self):
        self.nodes = {}
        self.vehicle = Vehicle()
        self.initialize_nodes()
        self.setup_visualization()
        
    def initialize_nodes(self):
        # Simulate user input for 15 nodes
        np.random.seed(42)  # For reproducibility
        for i in range(15):
            x = np.random.uniform(0, 50)  # Random x coordinate
            y = np.random.uniform(0, 50)  # Random y coordinate
            is_priority = i + 1 in [5, 7]  # 5th and 7th nodes are priority
            self.nodes[i + 1] = Node(
                id=i + 1,
                x=x,
                y=y,
                is_priority=is_priority,
                status='waiting'
            )
    
    def setup_visualization(self):
        # Set up the figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-5, 55)
        self.ax.set_ylim(-5, 55)
        self.ax.set_title('Autonomous Vehicle Routing Visualization')
        self.ax.set_xlabel('X coordinate (km)')
        self.ax.set_ylabel('Y coordinate (km)')
        
        # Create the vehicle marker
        self.vehicle_marker, = self.ax.plot([0], [0], 'bo', markersize=10, label='Vehicle')
        
        # Plot all nodes
        for node_id, node in self.nodes.items():
            color = 'red' if node.is_priority else 'green'
            marker = '^' if node.is_priority else 'o'
            size = 12 if node.is_priority else 8
            self.ax.plot(node.x, node.y, marker=marker, color=color, markersize=size)
            self.ax.annotate(f"{node_id}", (node.x, node.y), 
                            textcoords="offset points", 
                            xytext=(0,5), 
                            ha='center')
        
        # Create path line
        self.path_line, = self.ax.plot([0], [0], 'b-', alpha=0.5, label='Vehicle Path')
        
        # Create a legend
        self.ax.plot([], [], 'ro', markersize=8, label='Priority Node')
        self.ax.plot([], [], 'go', markersize=8, label='Regular Node')
        self.ax.legend(loc='upper right')
        
        # Create status text
        self.status_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                       fontsize=9, verticalalignment='top', 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.update_status_text()
        
        plt.tight_layout()
        
    def update_status_text(self):
        status = f"Total Distance: {self.vehicle.total_distance:.2f} km\n"
        status += f"Time Elapsed: {self.vehicle.time_elapsed:.2f} h\n"
        status += f"Current Passengers: {self.vehicle.passengers}\n"
        status += f"Position: ({self.vehicle.current_position[0]:.2f}, {self.vehicle.current_position[1]:.2f})"
        self.status_text.set_text(status)
    
    def update_visualization(self, wait=True):
        # Update vehicle position
        self.vehicle_marker.set_data([self.vehicle.current_position[0]], [self.vehicle.current_position[1]])
        
        # Update path line
        path_x = [point[0] for point in self.vehicle.path_history]
        path_y = [point[1] for point in self.vehicle.path_history]
        self.path_line.set_data(path_x, path_y)
        
        # Update node colors based on status
        self.ax.clear()
        self.ax.set_xlim(-5, 55)
        self.ax.set_ylim(-5, 55)
        self.ax.set_title('Autonomous Vehicle Routing Visualization')
        self.ax.set_xlabel('X coordinate (km)')
        self.ax.set_ylabel('Y coordinate (km)')
        
        for node_id, node in self.nodes.items():
            if node.status == 'waiting':
                color = 'red' if node.is_priority else 'green'
            elif node.status == 'picked':
                color = 'blue'
            else:  # dropped
                color = 'gray'
                
            marker = '^' if node.is_priority else 'o'
            size = 12 if node.is_priority else 8
            self.ax.plot(node.x, node.y, marker=marker, color=color, markersize=size)
            self.ax.annotate(f"{node_id}", (node.x, node.y), 
                            textcoords="offset points", 
                            xytext=(0,5), 
                            ha='center')
                            
        # Re-add vehicle marker and path
        self.vehicle_marker, = self.ax.plot([self.vehicle.current_position[0]], [self.vehicle.current_position[1]], 
                                           'bo', markersize=10, label='Vehicle')
        self.path_line, = self.ax.plot(path_x, path_y, 'b-', alpha=0.5, label='Vehicle Path')
        
        # Re-add legend
        self.ax.plot([], [], 'ro', markersize=8, label='Priority Waiting')
        self.ax.plot([], [], 'go', markersize=8, label='Regular Waiting')
        self.ax.plot([], [], 'bo', markersize=8, label='Picked Up')
        self.ax.plot([], [], 'o', color='gray', markersize=8, label='Dropped')
        self.ax.legend(loc='upper right')
        
        # Update status text
        self.status_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                       fontsize=9, verticalalignment='top', 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.update_status_text()
        
        plt.draw()
        plt.pause(0.01)
        if wait:
            plt.pause(0.5)  # Add a small delay to see the movement

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def find_nearest_priority_node(self) -> Node:
        priority_nodes = [node for node in self.nodes.values() 
                         if node.is_priority and node.status == 'waiting']
        if not priority_nodes:
            return None
        
        return min(priority_nodes, 
                  key=lambda n: self.calculate_distance(
                      self.vehicle.current_position, 
                      (n.x, n.y)
                  ))

    def find_optimal_next_node(self) -> Node:
        waiting_nodes = [node for node in self.nodes.values() 
                        if node.status == 'waiting']
        if not waiting_nodes:
            return None
        
        # First check priority nodes
        priority_node = self.find_nearest_priority_node()
        if priority_node:
            return priority_node

        # If no priority nodes, find the nearest regular node
        return min(waiting_nodes, 
                  key=lambda n: self.calculate_distance(
                      self.vehicle.current_position, 
                      (n.x, n.y)
                  ))

    def execute_command(self, command: str, node_id: int = None):
        if command == "status":
            self.display_status()
        elif command == "pickup" and node_id:
            self.pickup_passenger(node_id)
        elif command == "drop" and node_id:
            self.drop_passenger(node_id)
        elif command == "auto":
            self.run_autonomous_mode()
        elif command == "visualize":
            plt.show(block=True)
        else:
            print("Invalid command")

    def pickup_passenger(self, node_id: int):
        if node_id not in self.nodes:
            print(f"Node {node_id} does not exist")
            return
        
        node = self.nodes[node_id]
        if node.status != 'waiting':
            print(f"Node {node_id} is not waiting for pickup")
            return

        print(f"Moving to pick up passenger at node {node_id}...")
        distance, time = self.vehicle.move_to((node.x, node.y))
        node.status = 'picked'
        self.vehicle.passengers.append(node_id)
        print(f"Picked up passenger at node {node_id}")
        print(f"Distance traveled: {distance:.2f} km")
        print(f"Time taken: {time:.2f} hours")
        
        # Update visualization
        self.update_visualization()

    def drop_passenger(self, node_id: int):
        if node_id not in self.vehicle.passengers:
            print(f"Passenger {node_id} is not in the vehicle")
            return

        node = self.nodes[node_id]
        node.status = 'dropped'
        self.vehicle.passengers.remove(node_id)
        print(f"Dropped passenger from node {node_id}")
        
        # Update visualization
        self.update_visualization()

    def run_autonomous_mode(self):
        self.update_visualization(wait=False)
        
        while True:
            next_node = self.find_optimal_next_node()
            if not next_node:
                break

            print(f"\nMoving to next optimal node: {next_node.id}")
            self.pickup_passenger(next_node.id)
            self.drop_passenger(next_node.id)

        print("\nAll nodes serviced!")
        print(f"Total distance: {self.vehicle.total_distance:.2f} km")
        print(f"Total time: {self.vehicle.time_elapsed:.2f} hours")
        
        # Final visualization update
        self.update_visualization(wait=False)

    def display_status(self):
        print("\nCurrent Status:")
        print(f"Vehicle position: ({self.vehicle.current_position[0]:.2f}, {self.vehicle.current_position[1]:.2f})")
        print(f"Total distance: {self.vehicle.total_distance:.2f} km")
        print(f"Current passengers: {self.vehicle.passengers}")
        print("\nNodes status:")
        for node_id, node in self.nodes.items():
            priority_str = " (PRIORITY)" if node.is_priority else ""
            print(f"Node {node_id}{priority_str}: ({node.x:.2f}, {node.y:.2f}) - {node.status}")
        
        # Update visualization
        self.update_visualization(wait=False)

# Example usage
def main():
    system = RoutingSystem()
    print("Welcome to Visual Autonomous Vehicle Routing System")
    print("Commands: status, pickup <node_id>, drop <node_id>, auto, visualize, exit")
    
    # Display initial visualization
    system.update_visualization(wait=False)
    
    while True:
        command = input("\nEnter command: ").strip().lower()
        if command == "exit":
            plt.close()
            break
            
        if command == "auto":
            system.execute_command("auto")
        elif command.startswith("pickup "):
            try:
                _, node_id = command.split()
                system.execute_command("pickup", int(node_id))
            except (ValueError, IndexError):
                print("Invalid pickup command. Format: pickup <node_id>")
        elif command.startswith("drop "):
            try:
                _, node_id = command.split()
                system.execute_command("drop", int(node_id))
            except (ValueError, IndexError):
                print("Invalid drop command. Format: drop <node_id>")
        elif command == "status":
            system.execute_command("status")
        elif command == "visualize":
            system.execute_command("visualize")
        else:
            print("Invalid command")

if __name__ == "__main__":
    main()