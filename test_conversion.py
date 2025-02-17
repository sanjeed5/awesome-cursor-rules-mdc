import os
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_conversion_status(source_folder: str, output_folder: str) -> dict:
    """
    Analyze the conversion status by comparing source and output folders.
    
    Args:
        source_folder: Path to the source folder containing original rules
        output_folder: Path to the output folder containing .mdc files
        
    Returns:
        dict: Report containing status of each folder and .mdc file counts
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    if not source_path.exists():
        raise ValueError(f"Source folder {source_folder} does not exist")
    
    report = {
        "total_source_folders": 0,
        "total_output_folders": 0,
        "total_mdc_files": 0,
        "folders": {}
    }
    
    # Get all source subdirectories
    source_subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    report["total_source_folders"] = len(source_subdirs)
    
    for source_subdir in source_subdirs:
        folder_name = source_subdir.name
        corresponding_output = output_path / folder_name
        
        folder_report = {
            "has_source": True,
            "has_output": corresponding_output.exists(),
            "has_cursorrules": (source_subdir / ".cursorrules").exists(),
            "mdc_files": [],
            "mdc_count": 0,
            "status": "unknown"
        }
        
        # Check for .mdc files if output folder exists
        if folder_report["has_output"]:
            mdc_files = list(corresponding_output.glob("*.mdc"))
            folder_report["mdc_files"] = [f.name for f in mdc_files]
            folder_report["mdc_count"] = len(mdc_files)
            report["total_mdc_files"] += folder_report["mdc_count"]
        
        # Determine status
        if not folder_report["has_cursorrules"]:
            folder_report["status"] = "no_source_rules"
        elif not folder_report["has_output"]:
            folder_report["status"] = "not_converted"
        elif folder_report["mdc_count"] == 0:
            folder_report["status"] = "converted_no_rules"
        else:
            folder_report["status"] = "converted"
        
        report["folders"][folder_name] = folder_report
    
    # Count total output folders
    report["total_output_folders"] = len([d for d in output_path.iterdir() if d.is_dir()])
    
    return report

def save_report(report: dict, output_file: str = "conversion_report.json"):
    """Save the analysis report to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {output_file}")

def main():
    """Main function to run the conversion analysis."""
    source_folder = "awesome-cursorrules/rules"
    output_folder = "awesome-cursor-rules-mdc"
    
    try:
        report = analyze_conversion_status(source_folder, output_folder)
        save_report(report)
        
        # Print summary
        logger.info(f"\nConversion Analysis Summary:")
        logger.info(f"Total source folders: {report['total_source_folders']}")
        logger.info(f"Total output folders: {report['total_output_folders']}")
        logger.info(f"Total .mdc files: {report['total_mdc_files']}")
        
        # Print status counts
        status_counts = {}
        for folder_data in report["folders"].values():
            status = folder_data["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info("\nStatus Breakdown:")
        for status, count in status_counts.items():
            logger.info(f"{status}: {count} folders")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 