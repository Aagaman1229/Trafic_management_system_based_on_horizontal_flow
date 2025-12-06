# main_per_frame.py - Count vehicles PER FRAME (not cumulative)
import cv2
import torch
import numpy as np
import time
import os
import csv

class PerFrameVehicleCounter:
    def __init__(self, video_path):
        print("🚗 PER-FRAME VEHICLE COUNTER 🚗")
        print("="*50)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Load YOLOv5 model
        print("\n🤖 Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.25  # Lower confidence for more detections
        self.model.classes = [1, 2, 3, 5, 7]  # Vehicles: bicycle, car, motorcycle, bus, truck
        print("✅ Model loaded!")
        
        # Counters - PER FRAME ONLY (not cumulative)
        self.frame_count = 0
        
        # Colors for different vehicles
        self.vehicle_colors = {
            'car': (0, 255, 0),        # Green
            'truck': (0, 255, 255),    # Yellow
            'bus': (255, 0, 0),        # Blue
            'motorcycle': (255, 0, 255), # Pink
            'bicycle': (0, 165, 255)    # Orange
        }
        
        # Vehicle names mapping
        self.vehicle_names = {
            2: 'car',
            7: 'truck',
            5: 'bus',
            3: 'motorcycle',
            1: 'bicycle'
        }
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Output video writer
        output_path = "results/output_per_frame.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # CSV file for per-frame data
        self.csv_file = open("results/per_frame_data.csv", "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame_number', 
            'total_vehicles', 
            'cars', 
            'trucks', 
            'buses', 
            'motorcycles', 
            'bicycles',
            'confidence_threshold'
        ])
        
        print(f"\n💾 Output: {output_path}")
        print(f"📊 CSV data: results/per_frame_data.csv")
        print("="*50)
    
    def draw_frame_stats(self, frame, frame_vehicles, vehicle_counts):
        """Draw PER-FRAME statistics on the frame"""
        # Create semi-transparent background for stats
        overlay = frame.copy()
        
        # Top-left panel for current frame stats
        panel_width = 350
        panel_height = 220
        panel_x = 20
        panel_y = 20
        
        cv2.rectangle(overlay, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        # Blend with original frame
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw border
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title - PER FRAME
        title = "CURRENT FRAME STATS"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        title_x = panel_x + (panel_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset = panel_y + 70
        line_height = 30
        
        # Frame number
        cv2.putText(frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                   (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence threshold
        cv2.putText(frame, f"Confidence: {self.model.conf:.2f}", 
                   (panel_x + 20, y_offset + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # BIG PER-FRAME TOTAL (Most Important!)
        per_frame_total = f"VEHICLES THIS FRAME: {frame_vehicles}"
        per_frame_size = cv2.getTextSize(per_frame_total, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        per_frame_x = panel_x + (panel_width - per_frame_size[0]) // 2
        cv2.putText(frame, per_frame_total, (per_frame_x, panel_y + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Vehicle breakdown for this frame
        breakdown_y = panel_y + 180
        breakdown_text = "Breakdown: "
        
        # Add vehicle types that were detected in this frame
        detected_vehicles = []
        for vehicle_type, count in vehicle_counts.items():
            if count > 0:
                detected_vehicles.append(f"{vehicle_type}:{count}")
        
        if detected_vehicles:
            breakdown_text += ", ".join(detected_vehicles)
            cv2.putText(frame, breakdown_text, (panel_x + 10, breakdown_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return frame
    
    def draw_vehicle_box(self, frame, box, vehicle_type, confidence):
        """Draw bounding box for a vehicle"""
        x1, y1, x2, y2 = map(int, box)
        color = self.vehicle_colors.get(vehicle_type, (0, 255, 0))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{vehicle_type} {confidence:.0%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Label background
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), 
                     color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame - count vehicles IN THIS FRAME ONLY"""
        self.frame_count += 1
        
        # Run detection
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # PER-FRAME counters (reset for each frame)
        frame_vehicle_count = 0
        frame_vehicle_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }
        
        # Process detections for THIS FRAME ONLY
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det[:6]
            
            if confidence > self.model.conf:
                class_id = int(class_id)
                
                # Get vehicle type
                if class_id in self.vehicle_names:
                    vehicle_type = self.vehicle_names[class_id]
                    
                    # Draw bounding box
                    frame = self.draw_vehicle_box(frame, [x1, y1, x2, y2], vehicle_type, confidence)
                    
                    # Update PER-FRAME counters
                    frame_vehicle_count += 1
                    frame_vehicle_counts[vehicle_type] += 1
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Draw PER-FRAME statistics
        frame = self.draw_frame_stats(frame, frame_vehicle_count, frame_vehicle_counts)
        
        # FPS display (small, at top-right)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (self.width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controls reminder (bottom)
        controls = "Q: Quit  |  P: Pause  |  +/-: Confidence  |  S: Screenshot"
        cv2.putText(frame, controls, (20, self.height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame, frame_vehicle_count, frame_vehicle_counts
    
    def run(self):
        """Main processing loop"""
        self.start_time = time.time()
        
        print("\n▶ STARTING PER-FRAME VEHICLE COUNTING")
        print("   Counting vehicles in EACH FRAME separately (not cumulative)")
        print("\n🎮 CONTROLS:")
        print("   Q = Quit")
        print("   P = Pause")
        print("   + = Increase confidence (fewer detections)")
        print("   - = Decrease confidence (more detections)")
        print("   S = Save screenshot")
        print("\n" + "-"*50)
        
        # Create resizable window
        window_name = '🚗 PER-FRAME VEHICLE COUNTER 🚗'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 700)
        
        # Track some statistics for final report
        all_frame_counts = []
        max_vehicles_in_frame = 0
        min_vehicles_in_frame = float('inf')
        frames_with_zero = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("\n🎬 End of video reached!")
                break
            
            # Process frame (PER FRAME counting)
            processed_frame, frame_vehicles, vehicle_counts = self.process_frame(frame)
            
            # Write to output video
            self.out.write(processed_frame)
            
            # Update window title with current frame info
            cv2.setWindowTitle(window_name, 
                              f'🚗 Frame {self.frame_count}: {frame_vehicles} vehicles 🚗')
            
            # Write PER-FRAME data to CSV
            self.csv_writer.writerow([
                self.frame_count,
                frame_vehicles,
                vehicle_counts['car'],
                vehicle_counts['truck'],
                vehicle_counts['bus'],
                vehicle_counts['motorcycle'],
                vehicle_counts['bicycle'],
                self.model.conf
            ])
            
            # Track statistics
            all_frame_counts.append(frame_vehicles)
            max_vehicles_in_frame = max(max_vehicles_in_frame, frame_vehicles)
            min_vehicles_in_frame = min(min_vehicles_in_frame, frame_vehicles)
            if frame_vehicles == 0:
                frames_with_zero += 1
            
            # Show progress every 30 frames
            if self.frame_count % 30 == 0:
                print(f"📊 Frame {self.frame_count}: {frame_vehicles} vehicles in this frame")
            
            # Display frame
            cv2.imshow(window_name, processed_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⏹ Stopping...")
                break
            elif key == ord('p'):
                print("⏸ Paused. Press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('+') or key == ord('='):
                self.model.conf = min(0.9, self.model.conf + 0.05)
                print(f"⬆ Confidence: {self.model.conf:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.model.conf = max(0.05, self.model.conf - 0.05)
                print(f"⬇ Confidence: {self.model.conf:.2f}")
            elif key == ord('s'):
                screenshot = f"results/frame_{self.frame_count}_count_{frame_vehicles}.jpg"
                cv2.imwrite(screenshot, processed_frame)
                print(f"📸 Screenshot saved: {screenshot}")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
        
        # Final summary
        self.show_final_summary(all_frame_counts, max_vehicles_in_frame, 
                              min_vehicles_in_frame, frames_with_zero)
    
    def show_final_summary(self, all_counts, max_count, min_count, zero_frames):
        """Show final summary of per-frame analysis"""
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        total_frames = len(all_counts)
        average_per_frame = np.mean(all_counts) if all_counts else 0
        median_per_frame = np.median(all_counts) if all_counts else 0
        
        print("\n" + "="*70)
        print("✅ PER-FRAME ANALYSIS COMPLETE")
        print("="*70)
        print(f"🎞 Total frames analyzed: {total_frames}")
        print(f"⏱ Processing time: {total_time:.2f} seconds")
        print(f"⚡ Average FPS: {total_frames/total_time:.2f}")
        
        print("\n📊 PER-FRAME VEHICLE STATISTICS:")
        print("-" * 40)
        print(f"  Average vehicles per frame: {average_per_frame:.2f}")
        print(f"  Median vehicles per frame: {median_per_frame:.2f}")
        print(f"  Maximum in one frame: {max_count}")
        print(f"  Minimum in one frame: {min_count}")
        print(f"  Frames with zero vehicles: {zero_frames} ({zero_frames/total_frames*100:.1f}%)")
        
        # Distribution analysis
        if all_counts:
            print(f"\n📈 VEHICLE DISTRIBUTION:")
            print("-" * 40)
            
            # Count frames with different vehicle counts
            count_distribution = {}
            for count in all_counts:
                count_distribution[count] = count_distribution.get(count, 0) + 1
            
            # Show most common counts
            sorted_counts = sorted(count_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            print("  Most common vehicle counts per frame:")
            for count, freq in sorted_counts:
                percentage = (freq / total_frames) * 100
                print(f"    {count} vehicles: {freq} frames ({percentage:.1f}%)")
        
        print(f"\n💾 Output files:")
        print(f"  Video: results/output_per_frame.mp4")
        print(f"  CSV data: results/per_frame_data.csv")
        
        # Create a simple summary CSV
        self.create_summary_csv(average_per_frame, median_per_frame, 
                              max_count, min_count, zero_frames, total_frames)
        
        print("="*70)
    
    def create_summary_csv(self, avg, median, max_val, min_val, zero_frames, total_frames):
        """Create a summary CSV file"""
        summary_path = "results/per_frame_summary.csv"
        with open(summary_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Statistic", "Value"])
            writer.writerow(["Total frames analyzed", total_frames])
            writer.writerow(["Average vehicles per frame", f"{avg:.2f}"])
            writer.writerow(["Median vehicles per frame", f"{median:.2f}"])
            writer.writerow(["Maximum vehicles in one frame", max_val])
            writer.writerow(["Minimum vehicles in one frame", min_val])
            writer.writerow(["Frames with zero vehicles", zero_frames])
            writer.writerow(["Percentage of zero-vehicle frames", f"{(zero_frames/total_frames*100):.1f}%"])
        
        print(f"  Summary: results/per_frame_summary.csv")

def main():
    print("\n" + "="*70)
    print("🚗 PER-FRAME VEHICLE COUNTING SYSTEM")
    print("="*70)
    print("This version counts vehicles SEPARATELY in each frame.")
    print("It does NOT add up vehicles across frames.")
    print("\nOutput includes:")
    print("  • Vehicles detected in EACH frame")
    print("  • CSV with per-frame counts")
    print("  • Statistics on vehicle distribution")
    print("="*70)
    
    # Video path
    video_path = "video1.mp4"
    
    if not os.path.exists(video_path):
        print(f"\n❌ Video '{video_path}' not found!")
        print("\nPlease place your video as 'video1.mp4'")
        return
    
    # Create and run counter
    counter = PerFrameVehicleCounter(video_path)
    if hasattr(counter, 'cap') and counter.cap:
        counter.run()

if __name__ == "__main__":
    main()