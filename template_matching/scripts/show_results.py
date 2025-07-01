#!/usr/bin/env python3
"""
Simple script to display the generated visualizations.
"""

import os
import sys
import subprocess
import platform

PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"

def show_image(image_path):
    """Open image with default system viewer."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        if platform.system() == "Linux":
            # Try different image viewers
            viewers = ['eog', 'display', 'feh', 'xdg-open']
            for viewer in viewers:
                try:
                    subprocess.run([viewer, image_path], check=True, capture_output=True)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', image_path], check=True)
            return True
        elif platform.system() == "Windows":
            subprocess.run(['start', image_path], shell=True, check=True)
            return True
    except Exception as e:
        print(f"⚠️  Could not open image: {e}")
        return False
    
    print("⚠️  No suitable image viewer found")
    return False

def list_available_visualizations():
    """List all available visualizations."""
    viz_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations')
    
    if not os.path.exists(viz_dir):
        print("❌ No visualizations found. Run visualize_results.py first.")
        return {}
    
    visualizations = {}
    
    # Landmark predictions
    pred_dir = os.path.join(viz_dir, 'landmark_predictions')
    if os.path.exists(pred_dir):
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
        visualizations['landmark_predictions'] = [(f, os.path.join(pred_dir, f)) for f in sorted(pred_files)]
    
    # Error analysis
    error_dir = os.path.join(viz_dir, 'error_analysis')
    if os.path.exists(error_dir):
        error_files = [f for f in os.listdir(error_dir) if f.endswith('.png')]
        visualizations['error_analysis'] = [(f, os.path.join(error_dir, f)) for f in sorted(error_files)]
    
    # Lung contours
    contour_dir = os.path.join(viz_dir, 'lung_contours')
    if os.path.exists(contour_dir):
        contour_files = [f for f in os.listdir(contour_dir) if f.endswith('.png')]
        visualizations['lung_contours'] = [(f, os.path.join(contour_dir, f)) for f in sorted(contour_files)]
    
    # Summary report
    summary_file = os.path.join(viz_dir, 'summary_report.txt')
    if os.path.exists(summary_file):
        visualizations['summary_report'] = summary_file
    
    return visualizations

def show_summary():
    """Display summary report."""
    viz_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations')
    summary_file = os.path.join(viz_dir, 'summary_report.txt')
    
    if os.path.exists(summary_file):
        print("\n" + "="*60)
        print("📊 TEMPLATE MATCHING RESULTS SUMMARY")
        print("="*60)
        
        with open(summary_file, 'r') as f:
            content = f.read()
        
        print(content)
    else:
        print("❌ Summary report not found")

def interactive_menu():
    """Show interactive menu for viewing results."""
    visualizations = list_available_visualizations()
    
    if not visualizations:
        print("❌ No visualizations found. Please run:")
        print("   python3 template_matching/scripts/visualize_results.py")
        return
    
    while True:
        print("\n" + "="*60)
        print("🎨 TEMPLATE MATCHING VISUALIZATION VIEWER")
        print("="*60)
        print("Choose what to view:")
        print()
        
        menu_options = []
        option_num = 1
        
        # Summary report
        if 'summary_report' in visualizations:
            print(f"{option_num}. 📋 Summary Report (text)")
            menu_options.append(('summary', visualizations['summary_report']))
            option_num += 1
        
        # Error analysis
        if 'error_analysis' in visualizations:
            print(f"{option_num}. 📊 Error Analysis")
            for name, path in visualizations['error_analysis']:
                print(f"   {option_num}.{chr(97 + len(menu_options) - option_num + 1)} {name}")
                menu_options.append(('image', path))
            option_num += 1
        
        # Landmark predictions
        if 'landmark_predictions' in visualizations:
            print(f"{option_num}. 🎯 Landmark Predictions")
            for name, path in visualizations['landmark_predictions']:
                print(f"   {option_num}.{chr(97 + len(menu_options) - option_num + 1)} {name}")
                menu_options.append(('image', path))
            option_num += 1
        
        # Lung contours
        if 'lung_contours' in visualizations:
            print(f"{option_num}. 🫁 Lung Contours")
            for name, path in visualizations['lung_contours']:
                print(f"   {option_num}.{chr(97 + len(menu_options) - option_num + 1)} {name}")
                menu_options.append(('image', path))
            option_num += 1
        
        print(f"{option_num}. 🔄 Show All Landmark Predictions")
        menu_options.append(('all_landmarks', None))
        option_num += 1
        
        print(f"{option_num}. 🔄 Show All Error Analysis")
        menu_options.append(('all_errors', None))
        option_num += 1
        
        print(f"{option_num}. ❌ Exit")
        
        print()
        try:
            choice = input("Enter your choice (number): ").strip()
            choice_num = int(choice) - 1
            
            if choice_num == len(menu_options):  # Exit
                break
            elif choice_num < 0 or choice_num >= len(menu_options):
                print("❌ Invalid choice")
                continue
            
            action_type, path = menu_options[choice_num]
            
            if action_type == 'summary':
                show_summary()
                input("\nPress Enter to continue...")
                
            elif action_type == 'image':
                print(f"🖼️  Opening: {os.path.basename(path)}")
                show_image(path)
                
            elif action_type == 'all_landmarks':
                if 'landmark_predictions' in visualizations:
                    print("🖼️  Opening all landmark prediction images...")
                    for name, path in visualizations['landmark_predictions']:
                        show_image(path)
                
            elif action_type == 'all_errors':
                if 'error_analysis' in visualizations:
                    print("🖼️  Opening all error analysis images...")
                    for name, path in visualizations['error_analysis']:
                        show_image(path)
            
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='View Template Matching Results')
    parser.add_argument('--summary', action='store_true', help='Show summary report only')
    parser.add_argument('--all-landmarks', action='store_true', help='Show all landmark predictions')
    parser.add_argument('--all-errors', action='store_true', help='Show all error analysis')
    
    args = parser.parse_args()
    
    if args.summary:
        show_summary()
        return
    
    visualizations = list_available_visualizations()
    
    if args.all_landmarks and 'landmark_predictions' in visualizations:
        print("🖼️  Opening all landmark prediction images...")
        for name, path in visualizations['landmark_predictions']:
            show_image(path)
        return
    
    if args.all_errors and 'error_analysis' in visualizations:
        print("🖼️  Opening all error analysis images...")
        for name, path in visualizations['error_analysis']:
            show_image(path)
        return
    
    # Default: interactive menu
    interactive_menu()

if __name__ == "__main__":
    main()