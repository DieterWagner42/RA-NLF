#!/usr/bin/env python3
"""
Test script to verify the enhanced CSV export includes flow data
"""

# Read the new CSV and check structure
import csv

csv_file = "UC3_Rocket_Launch_Improved_UC_Steps_RA_Classes.csv"

print("Analyzing enhanced CSV structure...")

try:
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        # Read header
        header = next(reader)
        print(f"\nCSV Header ({len(header)} columns):")
        for i, col in enumerate(header, 1):
            print(f"  {i:2}. {col}")
        
        # Count rows and check for flow data
        row_count = 0
        rows_with_control_flows = 0
        rows_with_data_flows = 0
        
        for row in reader:
            if len(row) >= 12:  # Ensure enough columns
                row_count += 1
                
                # Check for control flow data (columns 6, 7, 8)
                if row[6] or row[7] or row[8]:  # Control_Flow_Source, Type, Rule
                    rows_with_control_flows += 1
                
                # Check for data flow data (columns 9, 10, 11)  
                if row[9] or row[10] or row[11]:  # Data_Flow_Entity, Type, Description
                    rows_with_data_flows += 1
        
        print(f"\nCSV Statistics:")
        print(f"  Total rows: {row_count}")
        print(f"  Rows with control flow data: {rows_with_control_flows}")
        print(f"  Rows with data flow data: {rows_with_data_flows}")
        
        # Sample some rows
        print(f"\nSample rows:")
        with open(csv_file, 'r', encoding='utf-8') as f2:
            reader2 = csv.reader(f2, delimiter=';')
            next(reader2)  # Skip header
            
            for i, row in enumerate(reader2):
                if i >= 5:  # Show first 5 data rows
                    break
                if len(row) >= 12:
                    print(f"  Row {i+1}: {row[0]} | {row[2]} ({row[3]}) | CF: {row[6]} | DF: {row[9]} ({row[10]})")
        
        if rows_with_control_flows > 0 or rows_with_data_flows > 0:
            print(f"\nSUCCESS: CSV enhancement successful! Flow data is present.")
        else:
            print(f"\nWARNING: CSV structure enhanced but flow data is empty (may need flow analysis improvements)")
        
except FileNotFoundError:
    print(f"ERROR: CSV file not found: {csv_file}")
except Exception as e:
    print(f"ERROR: Error reading CSV: {e}")

print(f"\nCSV analysis completed")